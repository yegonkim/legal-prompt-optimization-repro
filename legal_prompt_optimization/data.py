"""Restore frozen cohorts from pinned public sources, verified by content hashes."""
import csv,json,re,tarfile,urllib.request
from pathlib import Path
from .core import digest,dump,read,rows,validate_rows
from .formatters import format_casehold,format_lar_echr,format_barexamqa,format_lexam

SOURCES={
 'ledgar':('coastalcph/lex_glue','c23fdff1a6bf74e0e1a71cb86f1e781d37da888c'),
 'casehold':('casehold/casehold','acc4532a67e0966e9ed7ec4ca543e8532f983b0c'),
 'lar_echr':('AUEB-NLP/lar-echr','3204a623238a3a589a595de58e1e78b7e11cde6b'),
 'barexamqa_closed_book':('reglab/barexam_qa','b7c245a7e4739f6c785faa2489e283c8daa449d1'),
 'lexam':('LEXam-Benchmark/LEXam','31e57ae395f92ed7284a4c28d278c55da892b898')}
SARA_URL='https://nlp.jhu.edu/law/sara/sara.tar.gz'
SARA_SHA='e4f1b845016fc38b95b702bbbb27a5c9af3eadfa856ba585662b8b2cd71c187c'

def input_digest(row):
    # Used for reconstruction instead of file-byte equality (metadata/JSON whitespace may differ).
    fields={k:row[k] for k in ['record_id','question','answer','public_labels']}
    if 'source_record_id' in row:fields['source_record_id']=row['source_record_id']
    if 'reference_feedback' in row:fields['reference_feedback']=row['reference_feedback']
    return digest(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':')))

def load_sources(task,entries,cache_dir,offline=False):
    cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
    if task=='sara_entailment':
        archive=cache_dir/'sara.tar.gz'
        if not archive.exists():
            if offline:raise FileNotFoundError('Offline SARA archive missing')
            urllib.request.urlretrieve(SARA_URL,archive)
        if digest(archive.read_bytes())!=SARA_SHA:raise ValueError('SARA archive checksum mismatch')
        return tarfile.open(archive)
    from huggingface_hub import hf_hub_download
    repo,revision=SOURCES[task];splits={r['metadata'].get('source_split','test') for r in entries};result={}
    if task=='lexam':
        from datasets import load_dataset,DownloadConfig
        for config in {e['metadata']['source_config'] for e in entries}:
            ds=load_dataset(repo,config,split='test',revision=revision,download_config=DownloadConfig(local_files_only=offline))
            result[config]={str(r['id']):dict(r) for r in ds}
        return result
    for split in splits:
        if task=='ledgar':
            import pyarrow.parquet as pq
            path=hf_hub_download(repo,f'ledgar/{split}-00000-of-00001.parquet',repo_type='dataset',revision=revision,local_files_only=offline)
            table=pq.read_table(path);labels=json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names']
            result[split]=(table.to_pylist(),labels)
        else:
            filename=f'data/all/{"val" if split=="validation" else split}.csv' if task=='casehold' else f'{split}.csv' if task=='lar_echr' else f'data/qa/{split}.csv'
            path=hf_hub_download(repo,filename,repo_type='dataset',revision=revision,local_files_only=offline)
            with open(path,newline='',encoding='utf-8') as handle:
                if task=='casehold':
                    reader=csv.reader(handle);next(reader);raw=[]
                    for r in reader:raw.append(dict(example_id=int(r[0]),citing_prompt=r[1],label=r[12],source_split=split,**{f'holding_{i}':r[i+2] for i in range(5)}))
                    result[split]={str(r['example_id']):r for r in raw}
                elif task=='lar_echr':result[split]=list(csv.DictReader(handle))
                else:result[split]={r['idx']:r for r in csv.DictReader(handle) if r['idx'] not in {'mbe_494','mbe_1121'}}
    return result

def restore_entry(task,entry,sources):
    m=dict(entry['metadata']);rid=m['record_id'];split=m.get('source_split','test');feedback=None
    if task=='ledgar':
        raw,labels=sources[split];r=raw[m.get('source_row_index',int(rid.rsplit(':',1)[1]))];answer=labels[r['label']]
        question=('Classify the main legal purpose of the following contract provision. Select exactly one of the public category names below.\n\nContract provision:\n'+r['text'].strip()+'\n\nPublic categories (all 100):\n'+' | '.join(labels))
        feedback='Dataset-designated category: '+answer+'. No authoritative rationale supplied.'
    elif task=='casehold':
        r=sources[split][str(m.get('source_record_id',rid.rsplit(':',1)[1]))];f=format_casehold(r);question=f['question'];answer=f['answer'];labels=list('ABCDE')
        feedback='Dataset-designated option: '+answer+'. No authoritative rationale supplied.'
    elif task=='lar_echr':
        index=int(rid.split(':',3)[2]);r=sources[split][index];f=format_lar_echr(r);question=f['question'];answer=f['answer'];labels=list('ABCD')
        feedback='Dataset-designated option: '+answer+'. No authoritative rationale supplied.'
    elif task=='barexamqa_closed_book':
        r=(list(sources[split].values())[m['source_row_index']] if 'source_row_index' in m else sources[split][m.get('source_record_id',rid.split(':',3)[-1])]);f=format_barexamqa(r,evidence=False);question=f['question'];answer=f['answer'];labels=list('ABCD');feedback=r['gold_passage']
    elif task=='lexam':
        config=m['source_config'];sid=rid.split(':',1)[1];r=sources[config][sid];f=format_lexam(r);question=f['question'];answer=f['answer'];labels=[str(i+1) for i in range(len(r['choices']) if isinstance(r['choices'],list) else int(config.split('_')[1]))]
        feedback='Dataset-designated option: '+answer+'. No authoritative rationale supplied.'
    else:
        cid=m.get('original_source_case_id',rid.split(':',1)[1]);case=sources.extractfile('sara/cases/'+cid+'.pl').read().decode()
        facts='\n'.join(x[2:] for x in case.split('% Text\n',1)[1].split('% Question',1)[0].splitlines() if x.startswith('% '))
        q=' '.join(x[2:] for x in case.split('% Question\n',1)[1].split('% Facts',1)[0].splitlines() if x.startswith('% '));statement,answer=q.rsplit(' ',1)
        sec=re.match(r's(\d+)_',cid).group(1);statute=sources.extractfile('sara/statutes/source/section'+sec).read().decode()
        question='Determine whether the statement is entailed under the supplied statute and facts. Return Entailment or Contradiction as your final answer.\n\nStatute:\n'+statute+'\n\nDescription:\n'+facts+'\n\nStatement: '+statement
        labels=['Entailment','Contradiction'];feedback='Original dataset designated answer: '+answer+'. No authoritative worked rationale supplied.'
    if task!='sara_entailment':question=question.removesuffix('Return only the correct option label as your final answer.').rstrip()
    row={**m,'question':question,'answer':answer,'public_labels':labels}
    if entry['has_feedback']:row['reference_feedback']=feedback
    if input_digest(row)!=entry['input_sha256']:raise ValueError(f'Restored input mismatch: {rid}; preserve source, do not silently substitute')
    return row

def prepare(manifest,out,cache_dir,offline=False):
    manifest=read(manifest);task=manifest['dataset'];out=Path(out)
    if out.exists():raise FileExistsError('No data overwrite')
    entries=[r for rs in manifest['splits'].values() for r in rs];source=load_sources(task,entries,cache_dir,offline);out.mkdir(parents=True)
    for split,records in manifest['splits'].items():
        data=[restore_entry(task,r,source) for r in records];validate_rows(data)
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    dump(out/'manifest.json',manifest)
    return {k:len(v) for k,v in manifest['splits'].items()}

def verify(manifest,directory,splits=None):
    m=read(manifest);result={}
    for split,entries in m['splits'].items():
        if splits is not None and split not in splits:continue
        data=rows(Path(directory)/(split+'.jsonl'));validate_rows(data)
        if len(data)!=len(entries):raise ValueError('Wrong cohort size')
        for r,e in zip(data,entries):
            if r['record_id']!=e['metadata']['record_id'] or input_digest(r)!=e['input_sha256']:raise ValueError('Wrong ordered cohort/input')
        result[split]=len(data)
    return result
