import json

### you can add few shot examples in the follow list (2 shot example), each sample is in a dict format with three keys as illustrated.

few_shot_examples = [
    {
        'context': 'dummydata',
        'question': 'dummyquestion',
        'answer': 'dummyanswer',
    },
    {
        'context': 'dummydata',
        'question': 'dummyquestion',
        'answer': 'dummyanswer',
    },
]

with open('./few_shot_example.jsonl', 'w') as f:
    for idx, item in enumerate(few_shot_examples):
        f.write(json.dumps(item) + '\n')
        