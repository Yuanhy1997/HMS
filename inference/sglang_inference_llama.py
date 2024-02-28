from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import os, sys
from tqdm import tqdm
import json
import sglang
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

with open(sys.argv[1], 'r') as f:
    questions = [json.loads(item) for item in f.readlines()]

PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"

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

