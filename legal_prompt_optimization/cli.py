"""Small CLI; imports no GPU libraries for help, data checks or bundle verification."""
import argparse,contextlib,fcntl,importlib.metadata,json,os,signal,subprocess,sys,time
from pathlib import Path
from .core import ACTORS,digest,dump,file_sha,read,rows,validate_rows
from .data import prepare,verify
from .engine import optimize,readout

def selected(path):
    value=read(path)
    return value.get('best_candidate',value.get('selected_program',value))
def generator_instruction(path):
    value=selected(path)
    if set(value)!={'instruction'}:raise ValueError('Expected a frozen response instruction')
    return value['instruction']
def validate_config(c):
    if c['mode'] not in ['response','judge','joint']:raise ValueError('Unknown mode')
    if c['student']['model']!='microsoft/Phi-4-mini-instruct':raise ValueError('All inference roles must use frozen Phi-4-mini')
    if c.get('module_selector','round_robin') not in ['round_robin','all']:raise ValueError('Unknown component selector')
    if c['mode']=='joint' and c['module_selector']=='all' and c['proposer_call_cap']%5:raise ValueError('Reserve complete five-proposal iterations')
    if c['actor_max_tokens']!=1024 or c['judge_max_tokens'] not in [512,1024]:raise ValueError('Unexpected output caps')

def check_versions():
    for name,expected in {'gepa':'0.1.4','vllm':'0.31.0','transformers':'5.17.0','xgrammar':'0.2.7','torch':'2.13.0'}.items():
        if importlib.metadata.version(name)!=expected:raise RuntimeError(f'{name} must be {expected}; create the documented environment')

@contextlib.contextmanager
def gpu_lease(gpu,lock_dir):
    if gpu not in [0,1]:raise ValueError('Only physical GPU0 or GPU1 allowed by this research configuration')
    lock_dir=Path(lock_dir);lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/f'gpu{gpu}.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU occupied; never interrupt another job')
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);yield

def main(argv=None):
    ap=argparse.ArgumentParser(description='Frozen Phi response/judge or joint GEPA optimization')
    sub=ap.add_subparsers(dest='command',required=True)
    for name in ['prepare','verify-data']:
        a=sub.add_parser(name);a.add_argument('--manifest',required=True);a.add_argument('--data',required=True)
        if name=='prepare':a.add_argument('--cache',default='.source-cache');a.add_argument('--offline',action='store_true')
    a=sub.add_parser('verify-bundle');a.add_argument('--root',default='.')
    a=sub.add_parser('validate');a.add_argument('--config',required=True);a.add_argument('--data',required=True)
    for command in ['optimize','evaluate']:
        a=sub.add_parser(command);a.add_argument('--config',required=True);a.add_argument('--data',required=True);a.add_argument('--out',required=True)
        a.add_argument('--gpu',required=True,type=int);a.add_argument('--offline',action='store_true');a.add_argument('--lock-dir',default='runs/.leases')
        a.add_argument('--generator',help='Frozen response winner for judge optimization/sequential readout')
        if command=='evaluate':a.add_argument('--winner',required=True);a.add_argument('--splits',nargs='+',default=['dev','test'])
    a=sub.add_parser('audit');a.add_argument('--run',required=True);a.add_argument('--offline',action='store_true')
    args=ap.parse_args(argv)
    if args.command=='prepare':print(json.dumps(prepare(args.manifest,args.data,args.cache,args.offline)));return
    if args.command=='verify-data':print(json.dumps(verify(args.manifest,args.data)));return
    if args.command=='verify-bundle':
        root=Path(args.root);checks=read(root/'CHECKSUMS.json')
        for f,h in checks.items():
            if file_sha(root/f)!=h:raise ValueError(f'Bundle file changed: {f}')
        print(json.dumps({'files_verified':len(checks)}));return
    if args.command=='audit':
        from transformers import AutoTokenizer
        from .audit import audit_run
        c=read(Path(args.run)/'provenance.json')['config'];tok=AutoTokenizer.from_pretrained(c['student']['model'],revision=c['student']['revision'],local_files_only=args.offline)
        print(json.dumps(audit_run(args.run,tok)));return
    config=read(args.config);validate_config(config)
    names=['fit','select'] if args.command in ['optimize','validate'] else args.splits
    if not set(names)<={'fit','select','dev','test'}:raise ValueError('Unknown split')
    if args.command=='evaluate' and set(names)&{'fit','select'}:raise ValueError('Report DEV/TEST separately from optimization')
    parts={k:rows(Path(args.data)/(k+'.jsonl')) for k in names}
    for data in parts.values():validate_rows(data)
    if 'fit' in parts and {r['record_id'] for r in parts['fit']}&{r['record_id'] for r in parts['select']}:raise ValueError('FIT/SELECT overlap')
    manifest=Path(args.data)/'manifest.json'
    if manifest.exists():verify(manifest,args.data,splits=names)
    if args.command=='validate':print(json.dumps({'config_valid':True,'rows':{k:len(v) for k,v in parts.items()},'model_calls':0}));return
    out=Path(args.out)
    if out.exists():raise FileExistsError('Never overwrite or silently resume an interrupted run')
    check_versions();out.mkdir(parents=True)
    provenance=dict(operation=args.command,config=config,config_path=str(Path(args.config).resolve()),config_sha256=file_sha(args.config),source_files={str(f.resolve()):file_sha(f) for f in Path(__file__).parent.glob('*.py')},data={k:dict(path=str((Path(args.data)/(k+'.jsonl')).resolve()),sha256=file_sha(Path(args.data)/(k+'.jsonl'))) for k in names},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    for key in ['generator','winner']:
        path=getattr(args,key,None)
        if path:provenance[key]=dict(path=str(Path(path).resolve()),sha256=file_sha(path))
    dump(out/'provenance.json',provenance);code=1
    try:
        with gpu_lease(args.gpu,args.lock_dir):
            from .backend import LocalModels
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(SystemExit('Frozen runtime cap')))
            signal.alarm(config['runtime_seconds'] if args.command=='optimize' else 14400)
            backend=LocalModels(config,proposer=args.command=='optimize',offline=args.offline)
            generator=generator_instruction(args.generator) if args.generator else None
            if args.command=='optimize':optimize(config,parts,backend,out,generator)
            else:readout(config,parts,backend,out,selected(args.winner),generator)
            code=0;signal.alarm(0)
    except BaseException as error:
        dump(out/'failure.json',dict(error=repr(error),preserve_partial=True,no_retry=True));raise
    finally:dump(out/'exit.json',dict(exit_code=code))
    from .audit import audit_run
    print(json.dumps(audit_run(out,backend.stok)))
