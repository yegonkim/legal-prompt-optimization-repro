"""Local pinned vLLM backend. Heavy dependencies are imported only at runtime."""
import os
from pathlib import Path
from .core import render, schema

class LocalModels:
    def __init__(self,config,proposer=True,offline=False):
        # Workers start fresh interpreters: make the same compiler fix available there.
        os.environ['LEGAL_PHI_EOS_ALIGNMENT']='1'
        runtime=str(Path(__file__).parent/'runtime')
        os.environ['PYTHONPATH']=runtime+os.pathsep+os.environ.get('PYTHONPATH','')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        import importlib.util
        spec=importlib.util.spec_from_file_location('legal_phi_alignment',Path(runtime)/'sitecustomize.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from transformers import AutoTokenizer
        from vllm import LLM
        self.config=config
        student=config['student'];editor=config['proposer']
        self.stok=AutoTokenizer.from_pretrained(student['model'],revision=student['revision'],local_files_only=offline)
        if proposer:
            self.rtok=AutoTokenizer.from_pretrained(editor['model'],revision=editor['revision'],local_files_only=offline)
            self.editor=LLM(model=editor['model'],revision=editor['revision'],tokenizer_revision=editor['revision'],
                dtype='bfloat16',max_model_len=32768,max_num_seqs=1,gpu_memory_utilization=.64,enforce_eager=True,
                limit_mm_per_prompt={'image':0,'video':0,'audio':0},seed=42)
        self.student=LLM(model=student['model'],revision=student['revision'],tokenizer_revision=student['revision'],
            dtype='bfloat16',max_model_len=8192,max_num_seqs=32 if proposer else 64,
            gpu_memory_utilization=.25 if proposer else .75,enforce_eager=True,seed=42)
    def generate(self,requests):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        import xgrammar
        prompts=[];params=[]
        for r in requests:
            q=render(self.stok,r['instruction'],r['row']['question'],r['joint'])
            if len(self.stok.encode(q))+r['cap']>8192:raise RuntimeError('Student context exceeds8192; no clipping/retry')
            grammar=xgrammar.Grammar.from_json_schema(schema(r['row']['public_labels']),any_whitespace=False,indent=None,separators=(',',':'))
            prompts.append(q);params.append(SamplingParams(temperature=.7 if r['sampled'] else 0,top_p=.8 if r['sampled'] else 1,
                top_k=20 if r['sampled'] else -1,max_tokens=r['cap'],seed=r['seed'],
                structured_outputs=StructuredOutputsParams(grammar=str(grammar))))
        outputs=self.student.generate(prompts,params,use_tqdm=False)
        if len(outputs)!=len(requests):raise RuntimeError('Wrong output count')
        result=[]
        for prompt,out in zip(prompts,outputs):
            if len(out.outputs)!=1:raise RuntimeError('Expected one real completion per request')
            c=out.outputs[0]
            result.append(dict(prompt=prompt,text=c.text,token_ids=list(c.token_ids),finish_reason=c.finish_reason,output_index=c.index))
        return result
    def reflect(self,prompt,call):
        from vllm import SamplingParams
        messages=[dict(role='user',content=prompt)] if isinstance(prompt,str) else prompt
        q=self.rtok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        if len(self.rtok.encode(q))+8192>32768:raise SystemExit('Proposer context exceeds32768; no clipping/retry')
        c=self.editor.generate([q],SamplingParams(temperature=.7,top_p=.95,top_k=20,max_tokens=8192,seed=6000+call),use_tqdm=False)[0].outputs[0]
        record=dict(prompt=q,text=c.text,token_ids=list(c.token_ids),finish_reason=c.finish_reason,seed=6000+call)
        final=c.text.rsplit('</think>',1)[-1].strip()
        return final,record
