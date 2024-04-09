'''
This file is only for one process.
'''

import argparse
import json
import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

T_stage_def = {
    'T0': 'There is no primary tumor on imaging',
    'T1': "Tumor size less than or equal to 3cm or Superficial spreading tumor in central airways or Minimally invasive adenocarcinoma",
    # 'T1a': "Tumor size less than or equal to 1cm or Superficial spreading tumor in central airways or Minimally invasive adenocarcinoma",
    # 'T1b': "Tumor greater than 1cm but less than or equal to 2cm",
    # 'T1c': "Tumor greater than 2cm but less than or equal to 3cm",
    'T2a': "Tumor size greater than 3cm to less than or equal to 4cm",
    'T2b': "Tumor size greater than 4cm to less than or equal to 5cm, or tumor involving: visceral pleura, main bronchus (not carina), atelectasis to hilum",
    'T3': "Tumor size from 5cm to 7cm or invading chest wall, pericardium, phrenic nerve; or separate tumor nodule(s) in the same lobe.",
    'T4': "Tumor size greater than 7cm or tumor invading: mediastinum, diaphragm, heart, great vessels, recurrent laryngeal nerve, carina, trachea, esophagus, spine; or tumor nodule(s) in a different ipsilateral lobe.",
}

N_stage_def = {
    'N0': "No mention of regional node metastasis",
    'N1': "Metastasis in ipsilateral pulmonary or hilar nodes",
    'N2': "Metastasis in ipsilateral mediastinal or subcarinal nodes", 
    'N3': "Metastasis in contralateral mediastinal, hilar, or supraclavicular nodes",
}

M_stage_def = {
    'M0': "No mention of distant metastasis",
    'M1a': "Malignant pleural or pericardial effusionz or pleural or pericardial nodules or separate tumor nodule(s) in a contralateral lobe",
    'M1b': "Single extrathoracic metastasis",
    'M1c': "Multiple extrathoracic metastases (1 or>1 organ)",
}
PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"
TENPLATE = '''
Given the electronic health record:
{note}

Please answer this question accordingly:

Identify the cancer classification according to the class definition below. 
{stage_def}
If the note does not contain enough information, give a reliable possible prediction among the following classes.
'''

QUESTION_TEMPLATE = [
    TENPLATE.format(note=note, stage_def=T_stage_def),
    TENPLATE.format(note=note, stage_def=N_stage_def),
    TENPLATE.format(note=note, stage_def=M_stage_def),
]

import re

def t_stage_parse(string):
    result = re.findall('(T0|T1|T2a|T2b|T3|T4)', string)
    if len(result) == 0:
        return None
    else:
        return result[0]

def n_stage_parse(string):
    result = re.findall('(N0|N1|N2|N3)', string)
    if len(result) == 0:
        return None
    else:
        return result[0]

def m_stage_parse(string):
    result = re.findall('(M0|M1a|M1b|M1c)', string)
    if len(result) == 0:
        return None
    else:
        return result[0]
PARSE_FUNCS = [t_stage_parse, n_stage_parse, m_stage_parse]


def run_eval(
    model_path,
    data_df,
    max_new_token,
    temperature,
    answer_path,
):


    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto"
        )

    prompts = []
    notes = data_df['Sentences'].tolist()
    for item in tqdm(notes):
        for q_temp in QUESTION_TEMPLATE:
            query = q_temp.format(note=item)
            text = PROMPT_TEMPLATE.format(query=query)
            prompts.append(text)

    outputs = []
    for item in tqdm(prompts):
        model_inputs = tokenizer([item], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )

        outputs.append(output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids))


    for i, output in enumerate(outputs):
        parse_func = PARSE_FUNCS[i%3]
        response = tokenizer.batch_decode(output, skip_special_tokens=True)
        output = response[0].strip()
        output_model = output
        output_stage = parse_func(output_model)

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps({'key': i, 'output': output, 'stage': stage}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
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

    data_df = pd.read_excel(args.question_file)

    run_eval(
        args.model_path,
        data_df,
        args.max_new_token,
        args.temperature,
        args.answer_file,
    )
