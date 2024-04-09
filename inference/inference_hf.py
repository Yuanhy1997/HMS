'''
This file is only for one process.
'''

import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"

def run_eval(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    temperature,
    tp_size,
):


    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = '<unk>'
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = '</s>'
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = '<s>'
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = '<unk>'
    # if len(special_tokens_dict) > 0:
    #     tokenizer.add_special_tokens(special_tokens_dict)
    #     tokenizer.save_pretrained(model_path)
    # try:
    #     model = LLM(model=model_path, tensor_parallel_size=tp_size)
    # except RecursionError:
    #     model = LLM(model=model_path, tokenizer_mode='slow', tensor_parallel_size=tp_size)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )
    print(f"RANK: {os.environ['RANK']} | NUM_REPLICAS: {os.environ['WORLD_SIZE']}")
    print(f"Output to {answer_file}")
    print(f"Num Questions: {len(questions)}")
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_token)
    
    prompts = []
    for item in tqdm(questions):
        
        text = PROMPT_TEMPLATE.format(query=item['query'])
        prompts.append(text)

    prompt_id_map = {prompt: idx for idx, prompt in enumerate(prompts)}

    # outputs = model.generate(prompts, sampling_params)
    outputs = []
    for item in tqdm(prompts):
        model_inputs = tokenizer([item], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
        outputs.append(output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids))

    for i, output in enumerate(outputs):
        # output_ids = output.outputs[0].token_ids
        question = questions[i]

        response = tokenizer.batch_decode(output, skip_special_tokens=True)

        output = response[0].strip()
        question[f'output_{model_id}'] = output

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(question, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=None,
        help="The output answer file.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    args = parser.parse_args()

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]
    
    tp_size = 1
    
    run_eval(
        args.model_path,
        args.model_id,
        questions,
        args.answer_file,
        args.max_new_token,
        args.temperature,
        tp_size,
    )
