"""Regression coverage for simple runs, baseline tables and optional verification."""
import contextlib,copy,json,os,sys,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from test_contracts import Backend,Tokenizer,config,row,program
from legal_prompt_optimization.core import dump,file_sha,read,render,rows,digest
from legal_prompt_optimization.engine import Adapter,readout
from legal_prompt_optimization.audit import audit_run
from legal_prompt_optimization.cli import validate_config,main

ROOT=Path(__file__).parents[1]
class BaselineBackend(Backend):
    def generate(self,requests):
        outputs=super().generate(requests)
        for req,out in zip(requests,outputs):
            if req.get('condition')=='ordinary':out['text']=json.dumps({'reasoning':'Synthetic ordinary error','answer':'B'})
        return outputs

def run_readout(directory,baselines=False,alias=False,joint=False):
    root=Path(directory);data=root/'data.jsonl';r=row();data.write_text(json.dumps(r)+'\n');out=root/'run';out.mkdir()
    c=config('joint' if joint else 'judge');c['ordinary_instruction']='Generator' if alias else 'Ordinary';b=BaselineBackend()
    winner=program() if joint else {'instruction':'Judge'}
    result=readout(c,{'dev':[r]},b,out,winner,None if joint else 'Generator',baselines)
    dump(out/'provenance.json',{'operation':'evaluate','config':c,'baseline_table':baselines,'winner_program':winner,
        'generator_program':{'instruction':'Generator'},'data':{'dev':{'path':str(data),'sha256':file_sha(data)}}});dump(out/'exit.json',{'exit_code':0})
    return out,c,b,result

class Improvements(unittest.TestCase):
    def test_optional_five_column_baselines_share_optimized_candidates(self):
        with tempfile.TemporaryDirectory() as t:
            out,c,b,r=run_readout(t,baselines=True)
            self.assertEqual(r['new_calls'],36);m=r['metrics']['dev']
            self.assertEqual([m[k] for k in ['ordinary_pass1','ordinary_vote4','optimized_pass1','optimized_vote4','optimized_judge']],[0,0,1,1,1])
            self.assertTrue((out/'table.md').exists());self.assertTrue(audit_run(out,b.stok)['audit_passed'])
            (out/'table.md').write_text('incorrect table')
            with self.assertRaisesRegex(ValueError,'Baseline table'):audit_run(out,b.stok)
    def test_identical_ordinary_optimized_prompt_uses_one_bank(self):
        with tempfile.TemporaryDirectory() as t:
            out,c,b,r=run_readout(t,baselines=True,alias=True)
            self.assertEqual(r['new_calls'],20);self.assertTrue(r['ordinary_equals_optimized'])
            self.assertEqual(r['metrics']['dev']['ordinary_pass1'],r['metrics']['dev']['optimized_pass1']);self.assertTrue(audit_run(out,b.stok)['audit_passed'])
    def test_advanced_audit_rejects_plausible_but_wrong_metric(self):
        with tempfile.TemporaryDirectory() as t:
            out,c,b,r=run_readout(t)
            r['metrics']['dev']['judge']=0.5;dump(out/'metrics.json',r)
            with self.assertRaisesRegex(ValueError,'Reported metrics'):audit_run(out,b.stok)
    def test_advanced_audit_binds_frozen_instruction_even_matching_fake_hash(self):
        for joint in [False,True]:
            with self.subTest(joint=joint),tempfile.TemporaryDirectory() as t:
                out,c,b,result=run_readout(t,joint=joint);rs=rows(out/'responses.jsonl');r=rs[0]
                r['instruction']='UNSELECTED INSTRUCTION';r['prompt_sha256']=digest(render(b.stok,r['instruction'],row()['question'],joint))
                (out/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
                with self.assertRaisesRegex(ValueError,'instruction'):audit_run(out,b.stok)
    def test_reject_unsupported_config_values_and_unknown_fields(self):
        base=read(ROOT/'configs/ledgar/response.json')
        for key,value in [('context',4096),('proposer_context',16384),('proposer_max_tokens',4096),
            ('candidate_selection_strategy','current_best'),('use_merge',True),('unknown_parameter',123)]:
            with self.subTest(key=key):
                c={**base,key:value}
                with self.assertRaises(ValueError):validate_config(c)
        for path in (ROOT/'configs').rglob('*.json'):
            if path.parent.name == 'simple':
                from legal_prompt_optimization.simple_experiment import SearchConfig
                for item in read(path)['search']: SearchConfig(**item)
            else: validate_config(read(path))
    def test_truncated_proposer_raw_evidence_is_preserved(self):
        class Truncated(Backend):
            def reflect(self,prompt,call):return 'partial',{'prompt':prompt,'text':'FULL RAW TRUNCATED TEXT','token_ids':[1,2,3],'finish_reason':'length'}
        with tempfile.TemporaryDirectory() as t:
            a=Adapter(config(),Truncated(),t)
            with self.assertRaisesRegex(RuntimeError,'Truncated proposer'):a.reflect('reflection prompt')
            record=rows(Path(t)/'reflector_calls.jsonl')[0];self.assertEqual(record['text'],'FULL RAW TRUNCATED TEXT');self.assertEqual(record['token_ids'],[1,2,3]);self.assertEqual(a.reflections,1)
    def test_abnormal_student_batch_preserves_all_returned_outputs(self):
        class Abnormal(Backend):
            def generate(self,requests):
                outs=super().generate(requests);outs[0]['finish_reason']='abort';return outs
        with tempfile.TemporaryDirectory() as t:
            a=Adapter(config('response'),Abnormal(),t)
            with self.assertRaisesRegex(RuntimeError,'raw batch preserved'):a.generate([a.request(row(),'response','Instruction',draw=i) for i in range(2)])
            self.assertEqual(len(rows(Path(t)/'responses.jsonl')),2);self.assertEqual(a.calls,2)
    def test_offline_applies_to_tokenizers_and_model_workers(self):
        from legal_prompt_optimization.backend import LocalModels
        tokenizer_calls=[];models=[]
        def tokenizer(*args,**kwargs):tokenizer_calls.append(kwargs);return Tokenizer()
        def model(**kwargs):models.append(kwargs);return object()
        modules={'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)),
            'vllm':SimpleNamespace(LLM=model)}
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda *_:None))
        with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,modules),\
            patch('importlib.util.spec_from_file_location',return_value=spec),patch('importlib.util.module_from_spec',return_value=SimpleNamespace()):
            LocalModels(read(ROOT/'configs/ledgar/response.json'),offline=True)
            self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1');self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'],'1')
            self.assertTrue(all(r['local_files_only'] for r in tokenizer_calls));self.assertEqual(len(models),2)

    def test_default_cli_runs_lightweight_checks_not_advanced_audit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);data=p/'data';data.mkdir();out=p/'run';cfg=ROOT/'configs/ledgar/response.json'
            for split in ['fit','select']:(data/(split+'.jsonl')).write_text(json.dumps(row(split))+'\n')
            def optimized(c,parts,backend,destination,generator):
                result={'best_candidate':{'instruction':'Winner'},'candidates':[{'instruction':'Winner'}],'best_idx':0,'val_scores':[0.5],'student_calls':1,'proposer_calls':0};dump(Path(destination)/'winner.json',result);return result
            with patch('legal_prompt_optimization.cli.check_versions'),patch('legal_prompt_optimization.cli.gpu_lease',return_value=contextlib.nullcontext()),\
                patch('legal_prompt_optimization.backend.LocalModels',return_value=Backend()),patch('legal_prompt_optimization.cli.optimize',side_effect=optimized),\
                patch('legal_prompt_optimization.audit.audit_run',side_effect=AssertionError('Advanced audit must be optional')):
                main(['optimize','--config',str(cfg),'--data',str(data),'--out',str(out),'--gpu','0'])
            self.assertTrue(read(out/'checks.json')['checks_passed']);self.assertFalse((out/'audit.json').exists())
            p=read(out/'provenance.json');self.assertTrue(any(name.endswith('/runtime/sitecustomize.py') for name in p['source_files']))

if __name__=='__main__':unittest.main()
