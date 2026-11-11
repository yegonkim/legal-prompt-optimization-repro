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
    """A small supported configuration surface; reject everything we cannot honor."""
    required={'dataset','mode','student','proposer','actor_max_tokens','judge_max_tokens','ordinary_instruction',
        'candidate_selection_strategy','use_merge','context','proposer_context','proposer_max_tokens',
        'seed','reflection_minibatch_size','module_selector','student_call_cap','proposer_call_cap','runtime_seconds'}
    allowed=required|{'seed_instruction'}
    missing=required-set(c);unknown=set(c)-allowed
    if missing or unknown:raise ValueError(f'Config keys: missing={sorted(missing)}, unsupported={sorted(unknown)}')
    if c['mode'] not in ['response','judge','joint']:raise ValueError('Unknown mode')
    pinned={'student':('microsoft/Phi-4-mini-instruct','cfbefacb99257ffa30c83adab238a50856ac3083'),
        'proposer':('google/gemma-4-12B-it','707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')}
    for role,(model,revision) in pinned.items():
        if c[role]!={'model':model,'revision':revision}:raise ValueError(f'{role}: only the bundled frozen model/revision is supported')
    for key,value in {'candidate_selection_strategy':'pareto','context':8192,'proposer_context':32768,'proposer_max_tokens':8192}.items():
        if type(c[key]) is not type(value) or c[key]!=value:raise ValueError(f'Unsupported {key}: expected {value!r}')
    if c['use_merge'] is not False:raise ValueError('use_merge must be false')
    if c['module_selector'] not in ['round_robin','all']:raise ValueError('Unknown component selector')
    if c['mode']!='joint' and c['module_selector']!='round_robin':raise ValueError('Single-prompt stages support round_robin only')
    if c['mode']=='joint' and 'seed_instruction' in c:raise ValueError('Joint mode starts from ordinary_instruction plus the neutral judge')
    for key in ['reflection_minibatch_size','student_call_cap','proposer_call_cap','runtime_seconds']:
        if type(c[key]) is not int or c[key]<1:raise ValueError(f'{key} must be a positive integer')
    if type(c['seed']) is not int or not 0<=c['seed']<2**31:raise ValueError('seed must be a nonnegative32-bit integer')
    if c['mode']=='joint' and c['module_selector']=='all' and c['proposer_call_cap']%5:raise ValueError('Reserve complete five-proposal iterations')
    if type(c['actor_max_tokens']) is not int or type(c['judge_max_tokens']) is not int or c['actor_max_tokens']!=1024 or c['judge_max_tokens'] not in [512,1024]:raise ValueError('Unsupported output caps')
    for key in ['ordinary_instruction','seed_instruction']:
        if key in c and (type(c[key]) is not str or not c[key].strip()):raise ValueError(f'{key} must be a nonempty instruction')
    if type(c['dataset']) is not str or not c['dataset'].strip():raise ValueError('dataset must be a name')

def check_run_outputs(operation,out,config,result):
    """Cheap automatic checks; the optional advanced audit replays the full history."""
    if operation=='optimize':
        if result['best_candidate']!=result['candidates'][result['best_idx']] or result['val_scores'][result['best_idx']]!=max(result['val_scores']):raise RuntimeError('SELECT winner is inconsistent')
        if result['student_calls']>config['student_call_cap'] or result['proposer_calls']>config['proposer_call_cap']:raise RuntimeError('Budget exceeded')
    else:
        if any(not 0<=m[k]<=1 for m in result['metrics'].values() for k in ['individual','vote4','judge']):raise RuntimeError('Metric outside [0,1]')
    dump(Path(out)/'checks.json',{'checks_passed':True,'checks':'online prompt, seed, output-count, grading, budget and completion checks','advanced_audit_performed':False})

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
        if command=='evaluate':
            a.add_argument('--winner',required=True);a.add_argument('--splits',nargs='+',default=['dev','test'])
            a.add_argument('--baselines',action='store_true',help='Also evaluate ordinary response pass@1/vote@4 and write the five-column table')
            a.add_argument('--runtime-seconds',type=int,default=14400,help='Evaluation time cap, default4hours; optimization uses its config cap')
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
    if manifest.exists():
        if read(manifest)['dataset']!=config['dataset']:raise ValueError('Config dataset and manifest do not match')
        verify(manifest,args.data,splits=names)
    if args.command=='validate':print(json.dumps({'config_valid':True,'rows':{k:len(v) for k,v in parts.items()},'model_calls':0}));return
    out=Path(args.out)
    if out.exists():raise FileExistsError('Never overwrite or silently resume an interrupted run')
    baselines=bool(getattr(args,'baselines',False))
    if args.command=='evaluate' and args.runtime_seconds<1:raise ValueError('--runtime-seconds must be positive')
    if baselines and config['mode']=='joint':raise ValueError('--baselines is for response + judge evaluation')
    generator=generator_instruction(args.generator) if args.generator else None
    winner=selected(args.winner) if args.command=='evaluate' else None
    if args.command=='optimize' and config['mode']=='judge' and not generator:raise ValueError('Judge optimization requires --generator')
    if args.command=='evaluate':
        expected=set(ACTORS)|{'aggregator'} if config['mode']=='joint' else {'instruction'}
        if type(winner) is not dict or set(winner)!=expected or any(type(v)is not str or not v.strip() for v in winner.values()):raise ValueError('Winner has wrong prompt components')
        if config['mode']!='joint' and not generator:raise ValueError('Sequential evaluation requires --generator')
    check_versions();out.mkdir(parents=True)
    provenance=dict(operation=args.command,config=config,config_path=str(Path(args.config).resolve()),config_sha256=file_sha(args.config),source_files={str(f.resolve()):file_sha(f) for f in Path(__file__).parent.rglob('*.py')},data={k:dict(path=str((Path(args.data)/(k+'.jsonl')).resolve()),sha256=file_sha(Path(args.data)/(k+'.jsonl'))) for k in names},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),baseline_table=baselines,runtime_seconds=config['runtime_seconds'] if args.command=='optimize' else args.runtime_seconds)
    for key in ['generator','winner']:
        path=getattr(args,key,None)
        if path:provenance[key]=dict(path=str(Path(path).resolve()),sha256=file_sha(path))
    dump(out/'provenance.json',provenance);code=1
    try:
        with gpu_lease(args.gpu,args.lock_dir):
            from .backend import LocalModels
            signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(SystemExit('Frozen runtime cap')))
            signal.alarm(provenance['runtime_seconds'])
            backend=LocalModels(config,proposer=args.command=='optimize',offline=args.offline)
            if args.command=='optimize':result=optimize(config,parts,backend,out,generator)
            else:result=readout(config,parts,backend,out,winner,generator,baselines)
            check_run_outputs(args.command,out,config,result)
            code=0;signal.alarm(0)
    except BaseException as error:
        dump(out/'failure.json',dict(error=repr(error),preserve_partial=True,no_retry=True));raise
    finally:dump(out/'exit.json',dict(exit_code=code))
    print(json.dumps({'completed':True,'checks_passed':True,'out':str(out),'result':'winner.json' if args.command=='optimize' else 'metrics.json'}))
