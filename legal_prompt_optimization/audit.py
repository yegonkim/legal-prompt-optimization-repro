"""Optional advanced replay: frozen instructions, full frames and recomputed results."""
import json
from pathlib import Path
from .core import ACTORS,digest,dump,file_sha,frame,grade,read,render,rows,seed_for
from .engine import summarize_trials,baseline_table_text

def _selected(value):
    return value.get('best_candidate',value.get('selected_program',value))

def audit_run(root,tokenizer):
    root=Path(root);p=read(root/'provenance.json')
    if read(root/'exit.json')['exit_code']!=0:raise ValueError('Cannot certify failed run')
    for f,h in p.get('source_files',{}).items():
        if file_sha(f)!=h:raise ValueError('Run source changed')
    if p.get('config_path') and file_sha(p['config_path'])!=p['config_sha256']:raise ValueError('Config changed')
    for kind in ['winner','generator']:
        if kind in p and file_sha(p[kind]['path'])!=p[kind]['sha256']:raise ValueError('Frozen prompt source changed')
    c=p['config'];joint=c['mode']=='joint';parts={k:rows(v['path']) for k,v in p['data'].items()}
    for v in p['data'].values():
        if file_sha(v['path'])!=v['sha256']:raise ValueError('Dataset changed')
    original={r['record_id']:r for data in parts.values() for r in data}
    if len(original)!=sum(map(len,parts.values())):raise ValueError('Ambiguous record IDs across splits')
    raw=rows(root/'responses.jsonl');calls={r['call_id']:r for r in raw};expected={}
    if list(calls)!=list(range(1,len(raw)+1)):raise ValueError('Duplicate, missing or unordered call IDs')
    def frozen(kind):
        if kind in p:return _selected(read(p[kind]['path']))
        # Low-level CPU fixtures can carry a frozen program directly. CLI runs always hash its source file.
        key=kind+'_program'
        if key not in p:raise ValueError('Missing frozen '+kind+' instructions')
        return p[key]
    def bind(call_id,row,instruction,stage,trial,draw,seed_id=None,condition=None):
        if call_id not in calls:raise ValueError('Missing referenced call')
        r=calls[call_id];cap=c['judge_max_tokens'] if stage=='aggregator' else c['actor_max_tokens']
        rid=(row.get('source_record_id',row['record_id']) if joint else row['record_id']) if seed_id is None else seed_id
        value=(row['record_id'],row['question'],instruction,stage,trial,draw,seed_for(rid,draw),cap,joint,condition)
        if call_id in expected and expected[call_id]!=value:raise ValueError('Cache reused for different inputs or instructions')
        expected[call_id]=value
        if (r['record_id'],r['instruction'],r['stage'],r['trial'],r['draw'],r['seed'],r['cap'],r['joint'])!=(value[0],instruction,stage,trial,draw,value[6],cap,joint):
            raise ValueError('Response does not match frozen/evaluated instruction, role, draw or seed')
        if condition is not None and r.get('condition')!=condition:raise ValueError('Wrong evaluation condition')
        if r['sampled']!=(stage!='aggregator') or r.get('output_index',0)!=0:raise ValueError('Wrong sampling role/output index')
        if digest(render(tokenizer,instruction,row['question'],joint))!=r['prompt_sha256']:raise ValueError('Prompt changed or full reasoning omitted')
        if any(r[k]!=grade(r['text'],row)[k] for k in ['answer','valid','correct']):raise ValueError('Wrong grade')
        if len(r['token_ids'])>cap or r['finish_reason'] not in ['stop','length']:raise ValueError('Decoder violation')
        return r
    if p['operation']=='optimize':
        w=read(root/'winner.json')
        if w['best_candidate']!=w['candidates'][w['best_idx']] or w['val_scores'][w['best_idx']]!=max(w['val_scores']):raise ValueError('Winner not SELECT maximum')
        if len(raw)!=w['student_calls'] or len(raw)>c['student_call_cap']:raise ValueError('Student accounting')
        generator=w.get('generator_prefix');lookup=dict(original)
        if c['mode']=='judge':
            if 'generator' in p and frozen('generator')!={'instruction':generator}:raise ValueError('Generator prefix differs from frozen source')
            if not generator:raise ValueError('Missing generator prefix')
            for row in original.values():
                bank=[r for r in raw if r['phase']=='bank' and r['record_id']==row['record_id']]
                if len(bank)!=4 or {r['draw'] for r in bank}!=set(range(4)):raise ValueError('Incomplete judge bank')
                bank=sorted(bank,key=lambda r:r['draw'])
                for r in bank:bind(r['call_id'],row,generator,'response',0,r['draw'])
                f=frame(row,0,bank);lookup[f['record_id']]=f
        full_select={};selectids={r['record_id'] for r in parts['select']}
        if c['mode']=='judge':selectids={rid+'::aggregation-trial-0' for rid in selectids}
        for e in rows(root/'evaluations.jsonl'):
            if len(e['record_ids'])!=len(e['traces']):raise ValueError('Incomplete evaluation trace')
            candidate=e['candidate'];evaluated=[]
            expected_keys=set(ACTORS)|{'aggregator'} if joint else {'instruction'}
            if set(candidate)!=expected_keys:raise ValueError('Wrong evaluation program components')
            for rid,t in zip(e['record_ids'],e['traces']):
                if rid not in lookup:raise ValueError('Unknown evaluation row')
                row=lookup[rid];trial=t['trial']
                if joint:
                    if len(t['actor_calls'])!=4:raise ValueError('Incomplete actor trace')
                    actors=[bind(cid,row,candidate[k],k,trial,4*trial+i) for i,(k,cid) in enumerate(zip(ACTORS,t['actor_calls']))]
                    f=frame(row,trial,[dict(r,draw=4*trial+i) for i,r in enumerate(actors)])
                    out=bind(t['output_call'],f,candidate['aggregator'],'aggregator',trial,4*trial)
                else:
                    if t['actor_calls']:raise ValueError('Unexpected joint actors')
                    instruction=(generator+'\n\n' if c['mode']=='judge' else '')+candidate['instruction']
                    out=bind(t['output_call'],row,instruction,'aggregator' if c['mode']=='judge' else 'response',0,0)
                evaluated.append(out['correct'])
            if set(e['record_ids'])==selectids and len(e['record_ids'])==len(selectids):
                score=sum(evaluated)/len(evaluated);key=json.dumps(candidate,sort_keys=True)
                if key in full_select and full_select[key]!=score:raise ValueError('Inconsistent duplicate SELECT evaluations')
                full_select[key]=score
        if len(w['candidates'])!=len(w['val_scores']):raise ValueError('Incomplete candidate scores')
        for candidate,score in zip(w['candidates'],w['val_scores']):
            if full_select.get(json.dumps(candidate,sort_keys=True))!=score:raise ValueError('Recorded SELECT score differs from actual calls')
        refs=rows(root/'reflector_calls.jsonl') if (root/'reflector_calls.jsonl').exists() else []
        if [r['call'] for r in refs]!=list(range(1,len(refs)+1)) or len(refs)!=w['proposer_calls'] or len(refs)>c['proposer_call_cap']:raise ValueError('Proposer accounting')
        if any(r['finish_reason']!='stop' for r in refs):raise ValueError('Unfinished proposer in successful run')
        fitids={r['record_id'] for r in parts['fit']}
        if c['mode']=='judge':fitids={rid+'::aggregation-trial-0' for rid in fitids}
        for r in rows(root/'reflection_examples.jsonl') if (root/'reflection_examples.jsonl').exists() else []:
            if not set(r['record_ids'])<=fitids:raise ValueError('Non-FIT reflection')
    elif p['operation']=='evaluate':
        winner=frozen('winner');generator=None if joint else frozen('generator')['instruction']
        baselines=bool(p.get('baseline_table',False));alias=baselines and generator==c['ordinary_instruction']
        if joint and baselines:raise ValueError('Unsupported joint baseline table')
        metrics={}
        for split,data in parts.items():
            ordinary=[] if baselines else None;traces=[];judges=[]
            for row in data:
                optimized=[]
                if not joint:
                    conditions=[('ordinary',c['ordinary_instruction'])] if alias else ([('ordinary',c['ordinary_instruction'])] if baselines else [])+[('optimized',generator)]
                    for condition,instruction in conditions:
                        bank=[r for r in raw if r['record_id']==row['record_id'] and r['stage']=='response' and r.get('condition')==condition]
                        if len(bank)!=16 or {r['draw'] for r in bank}!=set(range(16)):raise ValueError('Incomplete selected actor bank')
                        bank=sorted(bank,key=lambda r:r['draw'])
                        for r in bank:bind(r['call_id'],row,instruction,'response',0,r['draw'],row.get('source_record_id',row['record_id']),condition)
                        if condition=='ordinary':ordinary+=bank
                        if condition=='optimized' or alias:optimized=bank
                for t in range(4):
                    if joint:
                        actors=[]
                        for i,k in enumerate(ACTORS):
                            found=[r for r in raw if r['record_id']==row['record_id'] and r['stage']==k and r['trial']==t]
                            if len(found)!=1:raise ValueError('Missing or duplicate joint actor')
                            actors.append(bind(found[0]['call_id'],row,winner[k],k,t,4*t+i))
                    else:actors=optimized[4*t:4*t+4]
                    f=frame(row,t,[dict(r,draw=4*t+i) for i,r in enumerate(actors)])
                    found=[r for r in raw if r['record_id']==f['record_id'] and r['stage']=='aggregator' and r['trial']==t]
                    if len(found)!=1:raise ValueError('Missing or duplicate judge')
                    instruction=winner['aggregator'] if joint else generator+'\n\n'+winner['instruction']
                    draw=4*t if joint else 860000+t;seed_id=f['source_record_id']
                    judge=bind(found[0]['call_id'],f,instruction,'aggregator',t,draw,seed_id,None if joint else 'judge')
                    judges.append(judge);traces.append(dict(row=row,trial=t,actors=actors,output=judge))
            metrics[split]=summarize_trials(data,traces,judges,ordinary,alias)
        reported=read(root/'metrics.json')
        expected_result=dict(metrics=metrics,new_calls=len(raw),full_four_reasonings=True,reporting_only=True,baseline_table=baselines,ordinary_equals_optimized=bool(alias))
        if reported!=expected_result:raise ValueError('Reported metrics differ from reconstructed candidate/judge outputs')
        if baselines and (root/'table.md').read_text()!=baseline_table_text(metrics):raise ValueError('Baseline table differs from recomputed metrics')
        expected_count=sum(map(len,parts.values()))*(36 if baselines and not alias else 20)
        if len(raw)!=expected_count:raise ValueError('Wrong actual response count')
    else:raise ValueError('Unknown operation')
    if set(expected)!=set(calls):raise ValueError('Unreferenced, extra or missing calls')
    result=dict(audit_passed=True,calls=len(raw),full_four_reasonings_verified=c['mode']!='response' or p['operation']=='evaluate',metrics_recomputed=True,
        instructions_bound=True,provenance_sha256=file_sha(root/'provenance.json'))
    dump(root/'audit.json',result);return result
