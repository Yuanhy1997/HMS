import json



raw_questions = [
    "Give three tips for staying healthy.",
    "What are the three primary colors?",
    "Describe the structure of an atom.",
    "How can we reduce air pollution?",
    "Describe a time when you had to make a difficult decision.",
    "Identify the odd one out.",
    "What is HMS?",
    "Introduce HMS"
]

raw_contexts = [
    "Give three tips for staying healthy.",
    "What are the three primary colors?",
    "Describe the structure of an atom.",
    "How can we reduce air pollution?",
    "Describe a time when you had to make a difficult decision.",
    "Identify the odd one out.",
    "What is HMS?",
    "Introduce HMS"
]



with open('./dummy_data.jsonl', 'w') as f:
    for idx, item in enumerate(raw_questions):
        f.write(json.dumps({'sample_idx': idx, 'context': raw_contexts[idx], 'question': item, 'answer': 'I don\'t know.'}) + '\n')
        