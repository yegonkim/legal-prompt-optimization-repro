"""Optional actual GEPA CPU integration; no models, downloads or GPU imports."""
import contextlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
from test_contracts import Backend,row,config
from legal_prompt_optimization.core import dump,file_sha,render
from legal_prompt_optimization.engine import optimize
from legal_prompt_optimization.audit import audit_run

class ImprovingBackend(Backend):
    def generate(self,requests):
        result=super().generate(requests)
        for req,out in zip(requests,result):
            if 'IMPROVED_ synthetic' not in req['instruction']:
                out['text']=json.dumps({'reasoning':'Synthetic incorrect answer.','answer':'B'})
        return result
    def reflect(self,prompt,call):
        return '```IMPROVED_ synthetic reusable reasoning instruction.```',dict(prompt=prompt,text='Synthetic proposer fixture',token_ids=[1],finish_reason='stop')

@unittest.skipUnless(importlib.util.find_spec('gepa'),'Install pinned runtime for actual GEPA integration')
class GEPAIntegration(unittest.TestCase):
    def test_all_three_modes(self):
        for mode in ['response','judge','joint']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                root=Path(t);out=root/'run';out.mkdir();parts={'fit':[row('fit')],'select':[row('select')]};c=config(mode);c['student_call_cap']=200;c['proposer_call_cap']=5 if mode=='joint' else 1;c['reflection_minibatch_size']=1
                for split,data in parts.items():(root/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))
                b=ImprovingBackend()
                with contextlib.redirect_stdout(io.StringIO()):f=optimize(c,parts,b,out,'IMPROVED_ generator' if mode=='judge' else None)
                self.assertEqual(max(f['val_scores']),1.0);self.assertGreater(f['best_idx'],0)
                dump(out/'provenance.json',{'operation':'optimize','config':c,'data':{s:{'path':str(root/(s+'.jsonl')),'sha256':file_sha(root/(s+'.jsonl'))} for s in parts}});dump(out/'exit.json',{'exit_code':0})
                self.assertTrue(audit_run(out,b.stok)['audit_passed'])
