"""Opt-in Phi XGrammar stop-ID correction; inherited by spawned workers."""
import os
if os.environ.get('LEGAL_PHI_EOS_ALIGNMENT')=='1':
    import xgrammar
    original=xgrammar.TokenizerInfo.from_huggingface
    def aligned(tokenizer,*,vocab_size=None,stop_token_ids=None):
        if tokenizer.eos_token_id==199999 and tokenizer.convert_tokens_to_ids('<|end|>')==200020:
            if stop_token_ids is not None:
                assert set(stop_token_ids if isinstance(stop_token_ids,list) else [stop_token_ids])<={199999,200020}
            stop_token_ids=[199999,200020]
        return original(tokenizer,vocab_size=vocab_size,stop_token_ids=stop_token_ids)
    xgrammar.TokenizerInfo.from_huggingface=staticmethod(aligned)
