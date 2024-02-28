from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import os, sys
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import sglang
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

with open(sys.argv[1], 'r') as f:
    questions = [json.loads(item) for item in f.readlines()]

if 'qwen' in sys.argv[2].lower():
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)
    _questions = []
    for item in tqdm(questions):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item['query']}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens_len = len(tokenizer(text + item['response'])['input_ids'])
        _questions.append(item)
        # if len(_questions) >= 100:
        #     break
    questions = _questions


    @sglang.function
    def text_qa(s, question):
        s += system("You are a helpful assistant.")
        s += user(question)
        s += assistant(gen("answer", max_tokens=2048))
    
    states = text_qa.run_batch(
        [
            {"question": item['query']} for item in questions
        ],
        progress_bar=True
    )

    # print(states)
    results = []

    for i, state in enumerate(states):
        for m in state.messages():
            results.append([m["role"], ":", m["content"]])
            
            # print(m["role"], ":", m["content"])

    
elif 'llama' in sys.argv[2]:

    PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], use_fast=False)

    @sglang.function
    def text_qa(s, question):
        s += PROMPT_TEMPLATE.format(query = question)
        s += gen("answer", max_tokens=2048)

    states = text_qa.run_batch(
        [
            {"question": item['query']} for item in questions
        ],
        progress_bar=True
    )

    results = []
    for i, state in enumerate(states):
        results.append({'Question': questions[i]['query'], 'Answer': state['answer']})

with open(sys.argv[3], 'w') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

