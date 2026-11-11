"""Replay output grading, native prompts, seeds, budgets and FIT-only feedback."""
from pathlib import Path
from .core import ACTORS,digest,dump,file_sha,frame,grade,read,render,rows,seed_for

def audit_run(root,tokenizer):
    root=Path(root);p=read(root/'provenance.json')
    if read(root/'exit.json')['exit_code']!=0:raise ValueError('Cannot certify failed run')
    for f,h in p.get('source_files',{}).items():
        if file_sha(f)!=h:raise ValueError('Run source changed')
    if p.get('config_path') and file_sha(p['config_path'])!=p['config_sha256']:raise ValueError('Config changed')
    for kind in ['winner','generator']:
        if kind in p and file_sha(p[kind]['path'])!=p[kind]['sha256']:raise ValueError('Frozen prompt source changed')
    config=p['config'];parts={k:rows(v['path']) for k,v in p['data'].items()}
    for v in p['data'].values():
        if file_sha(v['path'])!=v['sha256']:raise ValueError('Dataset changed')
    original={r['record_id']:r for data in parts.values() for r in data};data=dict(original)
    raw=rows(root/'responses.jsonl');calls={r['call_id']:r for r in raw}
    if len(calls)!=len(raw):raise ValueError('Duplicate calls')
    # Reconstruct every frame from exact complete raw actor texts, never parsed answers alone.
    groups={}
    if config['mode']=='judge' and p['operation']=='optimize':
        bank=[r for r in raw if r['phase']=='bank']
        for row in original.values():
            rs=[r for r in bank if r['record_id']==row['record_id']]
            if len(rs)!=4:raise ValueError('Incomplete judge bank')
            f=frame(row,0,rs);data[f['record_id']]=f
    if p['operation']=='evaluate':
        for split,rs in parts.items():
            for row in rs:
                for t in range(4):
                    actors=[r for r in raw if r['stage']!='aggregator' and r['record_id']==row['record_id'] and (r['trial']==t if config['mode']=='joint' else 4*t<=r['draw']<4*t+4)]
                    if len(actors)!=4:raise ValueError('Incomplete four-response trial')
                    actors=sorted(actors,key=lambda r:r['draw']);f=frame(row,t,actors);data[f['record_id']]=f
    elif config['mode']=='joint':
        for e in rows(root/'evaluations.jsonl'):
            for rid,t in zip(e['record_ids'],e['traces']):
                row=original[rid];actors=[dict(calls[c],draw=4*t['trial']+i) for i,c in enumerate(t['actor_calls'])]
                if [r['stage'] for r in actors]!=list(ACTORS):raise ValueError('Wrong actor order')
                f=frame(row,t['trial'],actors);groups[t['output_call']]=f
    checked=0
    for r in raw:
        row=groups.get(r['call_id'],data.get(r['record_id']))
        if row is None:raise ValueError('Unknown response row')
        q=render(tokenizer,r['instruction'],row['question'],r['joint'])
        if digest(q)!=r['prompt_sha256']:raise ValueError('Prompt changed or omitted candidate reasoning')
        rid=row.get('source_record_id',row['record_id']) if r['joint'] else row['record_id']
        if p['operation']=='evaluate' and not r['joint']:rid=row.get('source_record_id',row['record_id'])
        if r['seed']!=seed_for(rid,r['draw']):raise ValueError('Wrong source-ID seed')
        if any(r[k]!=grade(r['text'],row)[k] for k in ['answer','valid','correct']):raise ValueError('Wrong grade')
        if len(r['token_ids'])>r['cap'] or r['finish_reason'] not in ['stop','length']:raise ValueError('Decoder violation')
        checked+=1
    if p['operation']=='optimize':
        w=read(root/'winner.json')
        if w['best_candidate']!=w['candidates'][w['best_idx']] or w['val_scores'][w['best_idx']]!=max(w['val_scores']):raise ValueError('Winner not SELECT maximum')
        if checked>config['student_call_cap']:raise ValueError('Budget exceeded')
        refs=rows(root/'reflector_calls.jsonl') if (root/'reflector_calls.jsonl').exists() else []
        if len(refs)>config['proposer_call_cap'] or len(refs)!=w['proposer_calls']:raise ValueError('Proposer accounting')
        fitids={r['record_id'] for r in parts['fit']}
        if config['mode']=='judge':fitids={rid+'::aggregation-trial-0' for rid in fitids}
        for r in rows(root/'reflection_examples.jsonl') if (root/'reflection_examples.jsonl').exists() else []:
            if not set(r['record_ids'])<=fitids:raise ValueError('Non-FIT reflection')
    else:
        if checked!=20*sum(map(len,parts.values())):raise ValueError('Wrong actual response count')
    result=dict(audit_passed=True,calls=checked,full_four_reasonings_verified=config['mode']!='response',
        provenance_sha256=file_sha(root/'provenance.json'))
    dump(root/'audit.json',result);return result
