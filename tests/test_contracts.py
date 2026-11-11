import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from legal_prompt_optimization.core import *
from legal_prompt_optimization.engine import Adapter,BudgetExceeded,frozen_bank,readout
from legal_prompt_optimization.audit import audit_run
from legal_prompt_optimization.data import input_digest,verify

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return json.dumps(messages,ensure_ascii=False)
class Backend:
    def __init__(self):self.stok=Tokenizer();self.requests=[]
    def generate(self,requests):
        self.requests+=requests;outs=[]
        for r in requests:
            text=json.dumps({'reasoning':'Full reasoning with newline\nand "quoted" detail '+str(r['seed']),'answer':r['row']['answer']})
            outs.append(dict(prompt=render(self.stok,r['instruction'],r['row']['question'],r['joint']),text=text,token_ids=[1,2],finish_reason='stop',output_index=0))
        return outs

def row(rid='x',source='original'):
    return dict(record_id=rid,source_record_id=source,question='Synthetic example: choose the supported label. A. Alpha B. Beta',public_labels=['A','B'],answer='A',reference_feedback='Synthetic fixture feedback only.')
def config(mode='joint'):
    return dict(mode=mode,actor_max_tokens=1024,judge_max_tokens=512,student_call_cap=200,proposer_call_cap=10,
        reflection_minibatch_size=3,seed=42,module_selector='all')
def program():return {**{k:ORDINARY for k in ACTORS},'aggregator':NEUTRAL_JUDGE}

class Contracts(unittest.TestCase):
    def test_strict_parser(self):
        self.assertIsNone(parse_answer('{"reasoning":"x","answer":"A","answer":"B"}',['A','B']))
        self.assertIsNone(parse_answer('{"reasoning":"x","answer":"A","gold":"B"}',['A','B']))
        self.assertEqual(parse_answer('{"reasoning":"x","answer":"A"}',['A','B']),'A')
    def test_tie_first_appearance(self):self.assertEqual(vote(['B','A','A','B']),'B')
    def test_full_frame_escaping(self):
        rs=[dict(draw=d,text=json.dumps({'reasoning':'line\nquote "'+str(d),'answer':'A'})) for d in range(4)]
        f=frame(row(),0,rs);quoted=json.loads(f['question'].split('Four proposed responses, in original order:\n')[1])
        self.assertEqual([x['text'] for x in quoted],[r['text'] for r in rs])
    def test_gold_never_rendered(self):
        r=row();before=render(Tokenizer(),ORDINARY,r['question']);r.update(answer='POISON_GOLD',reference_feedback='POISON_FEEDBACK')
        self.assertEqual(before,render(Tokenizer(),ORDINARY,r['question']))
    def test_joint_native_leading_newlines(self):
        self.assertTrue(json.loads(render(Tokenizer(),ORDINARY,'Q',True))[0]['content'].startswith('\n\n'))
        self.assertFalse(json.loads(render(Tokenizer(),ORDINARY,'Q',False))[0]['content'].startswith('\n\n'))
    def test_independent_trials_and_source_seeds(self):
        with tempfile.TemporaryDirectory() as t:
            b=Backend();a=Adapter(config(),b,t);outputs,traces=a.program([row()],program(),range(4))
            self.assertEqual(a.calls,20);self.assertEqual(len(outputs),4)
            actors=[r for r in b.requests if r['stage']!='aggregator']
            self.assertEqual([r['seed'] for r in actors],[seed_for('original',i) for i in range(16)])
            judges=[r for r in b.requests if r['stage']=='aggregator'];self.assertEqual([r['seed'] for r in judges],[seed_for('x',4*t) for t in range(4)])
    def test_prefix_during_judge_bank_and_inference(self):
        with tempfile.TemporaryDirectory() as t:
            b=Backend();a=Adapter(config('judge'),b,t,prefix='GENERATOR PREFIX')
            parts=frozen_bank(a,{'fit':[row('f')],'select':[row('s')]},'GENERATOR PREFIX')
            self.assertEqual(a.calls,8);a.program(parts['fit'],{'instruction':'JUDGE SUFFIX'})
            self.assertEqual(b.requests[-1]['instruction'],'GENERATOR PREFIX\n\nJUDGE SUFFIX')
            self.assertIn('Full reasoning',b.requests[-1]['row']['question'])
    def test_budget_no_partial_call(self):
        with tempfile.TemporaryDirectory() as t:
            b=Backend();a=Adapter(config(),b,t,call_cap=3)
            with self.assertRaises(BudgetExceeded):a.program([row()],program())
            self.assertEqual(len(b.requests),0)
    def test_manifest_detects_membership_and_content_change(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);r=row();dump(p/'m.json',{'splits':{'fit':[{'metadata':{'record_id':'x'},'input_sha256':input_digest(r)}]}})
            (p/'fit.jsonl').write_text(json.dumps(r)+'\n');self.assertEqual(verify(p/'m.json',p),{'fit':1})
            r['question']='Changed';(p/'fit.jsonl').write_text(json.dumps(r)+'\n')
            with self.assertRaises(ValueError):verify(p/'m.json',p)
    def test_readout_replay_audit_and_corruption(self):
        for mode in ['joint','judge']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                p=Path(t);data=p/'data.jsonl';data.write_text(json.dumps(row())+'\n');out=p/'run';out.mkdir();c=config(mode);b=Backend()
                readout(c,{'dev':[row()]},b,out,program() if mode=='joint' else {'instruction':'Judge'},'Generator' if mode=='judge' else None)
                dump(out/'provenance.json',{'operation':'evaluate','config':c,'data':{'dev':{'path':str(data),'sha256':file_sha(data)}}});dump(out/'exit.json',{'exit_code':0})
                self.assertTrue(audit_run(out,b.stok)['audit_passed'])
                records=rows(out/'responses.jsonl');records[-1]['prompt_sha256']='corrupt';(out/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                with self.assertRaises(ValueError):audit_run(out,b.stok)
    def test_optimizer_validation_does_not_read_test(self):
        from legal_prompt_optimization.cli import main
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);r=row('fit');s=row('select')
            for k,value in [('fit',r),('select',s)]: (p/(k+'.jsonl')).write_text(json.dumps(value)+'\n')
            (p/'test.jsonl').write_text('INVALID DELIBERATE TEST POISON')
            m={'splits':{k:[{'metadata':{'record_id':v['record_id']},'input_sha256':input_digest(v)}] for k,v in [('fit',r),('select',s)]}}
            m['splits']['test']=[{'metadata':{'record_id':'test'},'input_sha256':'poison'}];dump(p/'manifest.json',m)
            c=read(Path(__file__).parents[1]/'configs/ledgar/response.json');dump(p/'config.json',c)
            main(['validate','--config',str(p/'config.json'),'--data',str(p)])

    def test_feedback_fit_only(self):
        with tempfile.TemporaryDirectory() as t:
            a=Adapter(config(),Backend(),t);a.fit_ids={'fit'}
            from types import SimpleNamespace
            with self.assertRaises(AssertionError):a.make_reflective_dataset(program(),SimpleNamespace(trajectories=[{'row':row('select')}]),list(program()))

if __name__=='__main__':unittest.main()
