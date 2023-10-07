import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template
import numpy as numpy


def run_eval(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    tp_size,
):

    model = LLM(model=model_path, tensor_parallel_size=tp_size)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_token)

    prompts = []
    for question in tqdm(questions):
        torch.manual_seed(0)
        conv = get_conversation_template(model_id)
        qs = question["instruction"]
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)

    prompt_id_map = {prompt: idx for idx, prompt in enumerate(prompts)}

    outputs = model.generate(prompts, sampling_params)

    for output in outputs:
        output_ids = output.outputs[0].token_ids
        question = questions[prompt_id_map[output.prompt]]

        # be consistent with the template's stop_token_ids
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = model.get_tokenizer().decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str and output.find(conv.stop_str) > 0:
            output = output[: output.find(conv.stop_str)]
        for special_token in model.get_tokenizer().special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()


        question['output'] = output
        question['generator'] = model_id

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(question) + "\n")

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
        "--question_file",
        type=str,
        default=None,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--answer_file",
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

    args = parser.parse_args()

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]
    
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] > 1:
        num_replicas = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        tp_size = torch.cuda.device_count() // num_replicas

        devices = ','.join([str(i) for i in range(rank*tp_size, (rank+1)*tp_size)])
        torch.cuda.set_device(devices)

        total_size = len(questions)
        questions = questions[rank:total_size:num_replicas]
        args.answer_file = args.answer_file.replace(".jsonl", f"_{rank}.jsonl")

        print(f"RANK: {rank} | NUM_REPLICAS: {num_replicas} | devices : {devices}")
    else:
        raise KeyError

    print(f"Output to {args.answer_file}")
    print(f"Num Questions: {len(questions)}")
    print(f"Conv Template: {get_conversation_template(args.model_id)}")

    run_eval(
        args.model_path,
        args.model_id,
        questions,
        args.answer_file,
        args.max_new_token,
        tp_size,
    )