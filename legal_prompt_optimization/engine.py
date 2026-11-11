"""Actual GEPA adapter: response, frozen-bank judge, or five-component joint."""
import json
import os
from pathlib import Path
from .core import ACTORS, JUDGE_SEED, ORDINARY, NEUTRAL_JUDGE, digest, dump, frame, grade, seed_for, vote

class BudgetExceeded(RuntimeError): pass

class Adapter:
    propose_new_texts=None
    def __init__(self,config,backend,out,mode=None,prefix='',call_cap=None):
        self.config=config;self.backend=backend;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.mode=mode or config['mode'];self.prefix=prefix;self.calls=0;self.search_calls=0;self.reflections=0;self.metrics=0
        self.call_cap=call_cap if call_cap is not None else config['student_call_cap'];self.cache={};self.fit_ids=set();self.phase='search'
    def record(self,name,value):
        with (self.out/name).open('a') as f:
            f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    def request(self,row,stage,instruction,trial=0,draw=None):
        joint=self.mode=='joint';sampled=stage!='aggregator';draw=(4*trial+(int(stage[-1]) if sampled and joint else 0)) if draw is None else draw
        rid=row.get('source_record_id',row['record_id']) if joint else row['record_id']
        key=(stage,rid,trial,instruction,digest(row['question']) if not sampled else '')
        return dict(row=row,stage=stage,instruction=instruction,trial=trial,draw=draw,seed=seed_for(rid,draw),
            sampled=sampled,joint=joint,cap=self.config['actor_max_tokens'] if sampled else self.config['judge_max_tokens'],key=key)
    def generate(self,requests,cache=False):
        if cache:
            pending={r['key']:r for r in requests if r['key'] not in self.cache};todo=list(pending.values())
        else:todo=requests
        if self.calls+len(todo)>self.call_cap:raise BudgetExceeded('Frozen physical student-call cap')
        fresh=[]
        for off in range(0,len(todo),64):
            chunk=todo[off:off+64];outputs=self.backend.generate(chunk)
            if len(outputs)!=len(chunk):raise RuntimeError('Wrong number of model outputs')
            for req,out in zip(chunk,outputs):
                self.calls+=1;self.search_calls+=self.phase=='search'
                item=dict(call_id=self.calls,record_id=req['row']['record_id'],source_record_id=req['row'].get('source_record_id'),
                    trial=req['trial'],draw=req['draw'],stage=req['stage'],instruction=req['instruction'],seed=req['seed'],
                    joint=req['joint'],sampled=req['sampled'],cap=req['cap'],phase=self.phase,
                    prompt_sha256=digest(out['prompt']),text=out['text'],token_ids=out['token_ids'],
                    finish_reason=out['finish_reason'],output_index=out.get('output_index',0),**grade(out['text'],req['row']))
                if len(item['token_ids'])>req['cap'] or item['finish_reason'] not in ['stop','length']:raise RuntimeError('Invalid decoder record')
                self.record('responses.jsonl',item);fresh.append(item)
                if cache:self.cache[req['key']]=item
            print(json.dumps(dict(stage=self.mode,phase=self.phase,student_calls=self.calls,proposer_calls=self.reflections)),flush=True)
        return [self.cache[r['key']] for r in requests] if cache else fresh
    def program(self,data,candidate,trials=(0,)):
        if self.mode=='joint':
            groups=[(row,t) for row in data for t in trials]
            requests=[self.request(row,k,candidate[k],t) for row,t in groups for k in ACTORS]
            actors=self.generate(requests,cache=True);traces=[];judges=[]
            for idx,(row,t) in enumerate(groups):
                rs=[dict(r,draw=4*t+i) for i,r in enumerate(actors[4*idx:4*idx+4])];f=frame(row,t,rs)
                judges.append(self.request(f,'aggregator',candidate['aggregator'],t))
                traces.append(dict(row=row,trial=t,actors=rs,frame=f))
            outputs=self.generate(judges,cache=True)
            for trace,output in zip(traces,outputs):trace['output']=output
            return outputs,traces
        instruction=candidate['instruction'];stage='aggregator' if self.mode=='judge' else 'response'
        if self.mode=='judge':instruction=self.prefix+'\n\n'+instruction
        requests=[self.request(row,stage,instruction,draw=0) for row in data]
        outputs=self.generate(requests)
        return outputs,[dict(row=row,output=out) for row,out in zip(data,outputs)]
    def evaluate(self,batch,candidate,capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        self.phase='search';outputs,traces=self.program(batch,candidate);self.metrics+=len(batch)
        self.record('evaluations.jsonl',dict(record_ids=[r['record_id'] for r in batch],candidate=candidate,
            traces=[dict(trial=t.get('trial',0),actor_calls=[r['call_id'] for r in t.get('actors',[])],output_call=t['output']['call_id']) for t in traces]))
        return EvaluationBatch(outputs=outputs,scores=[float(o['correct']) for o in outputs],trajectories=traces if capture_traces else None)
    def make_reflective_dataset(self,candidate,eval_batch,components_to_update):
        assert all(t['row']['record_id'] in self.fit_ids for t in eval_batch.trajectories),'SELECT cannot supply reflection feedback'
        examples=[]
        for t in eval_batch.trajectories:
            row=t['row'];out=t['output']
            if self.mode=='joint':
                ex={'Inputs':{'fixed_classifier_prefix':'','question':row['question'],'current_full_program':candidate},
                    'Four persona outputs':[r['text'] for r in t['actors']],'Aggregator output':out['text'],
                    'Feedback':{'end_to_end_correct':out['correct'],'correct_answer':row['answer'],
                        'reference_feedback':row.get('reference_feedback','Dataset label only.')}}
            elif self.mode=='judge':
                ex={'Inputs':{'fixed_classifier_prefix':self.prefix,'question_and_four_responses':row['question']},
                    'Generated response':out['text'],'Feedback':{'correct':out['correct'],'correct_answer':row['answer'],
                    'majority_answer':vote(row['candidate_answers']),'reference_feedback':row.get('reference_feedback','Dataset-designated label only.')}}
            else:ex={'Inputs':row['question'],'Generated response':out['text'],
                'Feedback':{'correct':out['correct'],'correct_answer':row['answer'],'reference_feedback':row['reference_feedback']}}
            examples.append(ex)
        self.record('reflection_examples.jsonl',dict(record_ids=[t['row']['record_id'] for t in eval_batch.trajectories],components=components_to_update,examples=examples))
        return {key:examples for key in components_to_update}
    def reflect(self,prompt):
        if self.reflections>=self.config['proposer_call_cap']:raise BudgetExceeded('Frozen proposer cap')
        self.reflections+=1
        final,record=self.backend.reflect(prompt,self.reflections)
        self.record('reflector_calls.jsonl',dict(call=self.reflections,**record));return final

def frozen_bank(adapter,parts,generator):
    adapter.phase='bank';frames={}
    for split,data in parts.items():
        bank=adapter.generate([adapter.request(row,'response',generator,draw=d) for row in data for d in range(4)])
        frames[split]=[frame(row,0,bank[4*i:4*i+4]) for i,row in enumerate(data)]
    adapter.phase='search';return frames

def optimize(config,parts,backend,out,generator=None):
    import gepa
    adapter=Adapter(config,backend,out,prefix=generator or '')
    if config['mode']=='judge':
        if not generator:raise ValueError('Judge optimization requires a frozen generator winner')
        parts=frozen_bank(adapter,parts,generator)
    adapter.fit_ids={r['record_id'] for r in parts['fit']};select=parts['select']
    if adapter.fit_ids&{r['record_id'] for r in select}:raise ValueError('FIT and SELECT IDs overlap')
    if config['mode']=='joint':seed={**{k:config.get('ordinary_instruction',ORDINARY) for k in ACTORS},'aggregator':NEUTRAL_JUDGE}
    else:seed={'instruction':config.get('seed_instruction',JUDGE_SEED if config['mode']=='judge' else ORDINARY)}
    def stop(_):
        reserve=(5*(len(select)+6) if config['mode']=='joint' else len(select)+2*config['reflection_minibatch_size'])
        return (adapter.reflections>=config['proposer_call_cap'] or adapter.calls+reserve>config['student_call_cap']
            or (config['mode']=='joint' and adapter.metrics+len(select)+6>config['student_call_cap']))
    metric_cap=config['student_call_cap']-(adapter.calls if config['mode']=='judge' else 0)
    result=gepa.optimize(seed_candidate=seed,trainset=parts['fit'],valset=select,adapter=adapter,reflection_lm=adapter.reflect,
        candidate_selection_strategy='pareto',module_selector=config.get('module_selector','round_robin'),
        reflection_minibatch_size=config['reflection_minibatch_size'],max_metric_calls=metric_cap,stop_callbacks=stop,
        use_merge=False,seed=config['seed'],run_dir=str(Path(out)/'gepa'),raise_on_exception=True)
    if result.total_metric_calls!=adapter.metrics:raise RuntimeError('GEPA metric accounting mismatch')
    freeze=dict(config=config,best_candidate=result.best_candidate,candidates=result.candidates,best_idx=result.best_idx,
        val_scores=result.val_aggregate_scores,student_calls=adapter.calls,proposer_calls=adapter.reflections,
        generator_prefix=generator,selected_on='SELECT only')
    dump(Path(out)/'winner.json',freeze);return freeze

def readout(config,parts,backend,out,winner,generator=None):
    joint=config['mode']=='joint';cap=sum(map(len,parts.values()))*20
    a=Adapter(config,backend,out,mode='joint' if joint else 'response',call_cap=cap);a.phase='readout';metrics={}
    for split,data in parts.items():
        if joint:judges,traces=a.program(data,winner,range(4))
        else:
            if not generator:raise ValueError('Sequential readout requires generator and judge winners')
            actor_requests=[a.request(row,'response',generator,draw=d) for row in data for d in range(16)]
            for request in actor_requests:request['seed']=seed_for(request['row'].get('source_record_id',request['row']['record_id']),request['draw'])
            actors=a.generate(actor_requests)
            requests=[];traces=[]
            for i,row in enumerate(data):
                for t in range(4):
                    rs=[dict(r,draw=4*t+j) for j,r in enumerate(actors[16*i+4*t:16*i+4*t+4])];f=frame(row,t,rs)
                    req=a.request(f,'aggregator',generator+'\n\n'+winner['instruction'],t,draw=860000+t)
                    # Historical sequential readout uses original source ID for the greedy judge seed.
                    req['seed']=seed_for(f['source_record_id'],860000+t);requests.append(req);traces.append(dict(row=row,trial=t,actors=rs,frame=f))
            judges=a.generate(requests)
            for t,j in zip(traces,judges):t['output']=j
        individual=[r for t in traces for r in t['actors']];n=len(data)
        metrics[split]=dict(questions=n,individual=sum(r['correct'] for r in individual)/(16*n),
            per_generator={k:sum(t['actors'][i]['correct'] for t in traces)/(4*n) for i,k in enumerate(ACTORS)},
            vote4=sum(vote([r['answer'] for r in t['actors']])==t['row']['answer'] for t in traces)/(4*n),
            judge=sum(j['correct'] for j in judges)/(4*n),invalid=sum(not r['valid'] for r in individual+judges),
            length_finishes=sum(r['finish_reason']=='length' for r in individual+judges))
    result=dict(metrics=metrics,new_calls=a.calls,full_four_reasonings=True,reporting_only=True)
    dump(Path(out)/'metrics.json',result);return result
