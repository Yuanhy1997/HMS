import argparse
import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from fastchat.model.model_adapter import model_adapters, register_model_adapter, BaseModelAdapter
from fastchat.conversation import conv_templates, register_conv_template, Conversation, SeparatorStyle, get_conv_template

few_shot_question_template = 'Given the contexts: {context}, please answer: {question} '
with open('./few_shot_example.jsonl', 'r') as f:
    incontext_few_shots = [json.loads(item) for item in f.readlines()]
messages = [[("Human", few_shot_question_template.format(context=item['context'], question=item["question"])), ("Assistant", item["answer"])] for item in incontext_few_shots]
messages = tuple(sum(messages, []))

register_conv_template(
    Conversation(
        name="custom_few_shot",
        system_message="A chat between a curious human and an artificial intelligence clinician. "
        "The assistant gives helpful and detailed answers to the human's questions related to biomedicine.",
        roles=("Human", "Assistant"),
        messages=messages,
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

register_conv_template(
    Conversation(
        name="pretrain_few_shot",
        system_message="",
        roles=("Question", "Answer"),
        messages=messages,
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence clinician. "
        "The assistant gives helpful and detailed answers to the human's questions related to biomedicine.",
        roles=("Human", "Assistant"),
        messages=messages[:2],
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    ),
    override=True
)

register_conv_template(
    Conversation(
        name="eevee",
        system_message="Below is a health record paired with a question that describes a task.  Write a response that appropriately answers the question.",
        roles=("### Health Record", "### Answer"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

class EeveeAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "eevee" in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("eevee")

class FewShotAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "fewshot" in model_path.lower() and "pretrain" not in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("custom_few_shot")

class PretrainFewShotAdapter(BaseModelAdapter):

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "pretrainfewshot" in model_path.lower()

    def get_default_conv_template(self, model_path: str):
        return get_conv_template("pretrain_few_shot")

register_model_adapter(EeveeAdapter)
register_model_adapter(FewShotAdapter)
register_model_adapter(PretrainFewShotAdapter)

from fastchat.model import get_conversation_template
import numpy as numpy
from transformers import AutoTokenizer

def run_eval(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    tp_size,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = '<pad>'
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = '</s>'
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = '<s>'
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = '<unk>'
    if len(special_tokens_dict) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.save_pretrained(model_path)
    try:
        model = LLM(model=model_path, tensor_parallel_size=tp_size)
    except RecursionError:
        model = LLM(model=model_path, tokenizer_mode='slow', tensor_parallel_size=tp_size)
    print('model loadeds')
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_token)

    prompts = []


    if conv.name=='eevee':
        question_template = '{context}\n\n### Question:\n{question}'
    else:
        question_template = few_shot_question_template
        
    for item in tqdm(questions):
        torch.manual_seed(0)
        if 'llama' in model_id.lower() and 'chat' not in model_id.lower():
            conv = get_conversation_template('pretrainfewshot')
        else:
            conv = get_conversation_template(model_id)
        qs = few_shot_question_template.format(context=item['context'], question=item["question"])
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

    args = parser.parse_args()

    with open(args.question_file, 'r') as f:
        questions = [json.loads(item) for item in f.readlines()]
    
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        num_replicas = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        tp_size = torch.cuda.device_count() // num_replicas
        devices = ','.join([str(i) for i in range(rank*tp_size, (rank+1)*tp_size)])
        # torch.cuda.set_device(devices)
        total_size = len(questions)
        questions = questions[rank:total_size:num_replicas]
        args.answer_file = args.answer_file.replace(".jsonl", f"_{rank}.jsonl")

        print(f"RANK: {rank} | NUM_REPLICAS: {num_replicas} | devices : {devices}")
    else:
        tp_size = 1

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
