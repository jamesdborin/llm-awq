from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
import torch
# eval imports
from lm_eval import evaluator
from awq.utils.lm_eval_adaptor import LMEvalAdaptor

from awq.entry import run_search, run_quant, run_eval

from awq.models.opt import OptAWQForCausalLM
from awq.models.t5 import FlanT5AWQForCausalLM
from awq.models.gptj import GPTJAWQForCausalLM

import os
from huggingface_hub import snapshot_download

def evalulate_model(model, tokenizer, model_name, tasks='wikitext', task_batch_size=1, task_n_shot=0):

    lm_eval_adaptor = LMEvalAdaptor(
        model_name = model_name,
        model = model, 
        tokenizer = tokenizer, 
        device='cuda', 
        batch_size=1
    )

    results = evaluator.simple_evaluate(
        model=lm_eval_adaptor,
        tasks=tasks.split(','),
        batch_size=task_batch_size,
        no_cache=True,
        num_fewshot=task_n_shot,
    )

    print(evaluator.make_table(results))

if __name__ == '__main__':
    hf_model_name = "EleutherAI/gpt-j-6b" # facebook/opt-125m 
    AWQ_CLASS = GPTJAWQForCausalLM # OptAWQForCausalLM 
    HF_CLASS = AutoModelForCausalLM # AutoModelForSeq2SeqLM
    MODEL_TYPE = 'opt'
    
    hf_local_save = f"hf_{hf_model_name.replace('/','_')}"
    awq_local_save = f"awq_{hf_model_name.replace('/','_')}"

    # download the original model
    ignore_patterns = ["*msgpack*", "*h5*"]
    ignore_patterns.append("*safetensors*")

    # Already downloaded
    snapshot_download(
        repo_id=hf_model_name, 
        local_dir=hf_local_save,
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_patterns
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_local_save)

    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 } 

    awq_model = AWQ_CLASS.from_pretrained(
        model_path = hf_local_save,
        model_type = MODEL_TYPE
    )


    awq_model.quantize(
        tokenizer = tokenizer,
        quant_config=quant_config, 
        run_search=True, 
        run_quant=True
    )

    awq_model.save_quantized(awq_local_save)

    # load the quantized model
    awq_model = AWQ_CLASS.from_quantized(
        model_path = awq_local_save, 
        model_filename = "awq_model_w4_g128.pt",
        model_type = MODEL_TYPE
    )

    # load hf model
    hf_model = HF_CLASS.from_pretrained(hf_local_save)

    # evaluate the original model
    # evalulate_model(
    #     model = hf_model, 
    #     tokenizer = tokenizer, 
    #     model_name = hf_model_name, 
    #     tasks='wikitext', 
    #     task_batch_size=1, 
    #     task_n_shot=0
    # )

    # evaluate the quantized model
    evalulate_model(        
        model = awq_model, 
        tokenizer = tokenizer, 
        model_name = hf_model_name, 
        tasks='wikitext', 
        task_batch_size=1, 
        task_n_shot=0
    )