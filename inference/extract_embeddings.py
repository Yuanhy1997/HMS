import transformers
import sys
import torch

model = transformers.AutoModelForCausalLM.from_pretrained(
        sys.argv[1],
        torch_dtype=torch.bfloat16,
    )
tokenizer = transformers.AutoTokenizer.from_pretrained(
        sys.argv[1],
        padding_side="right",
    )


input_texts = 'Trying to extract embeddings.'
input_ids = tokenizer(input_texts, return_tensors="pt").input_ids
with torch.no_grad():
     outputs = model(input_ids, return_dict = True, output_hidden_states = True)


final_layer_embedding = outputs.hidden_states[-1]
print(final_layer_embedding.shape)
