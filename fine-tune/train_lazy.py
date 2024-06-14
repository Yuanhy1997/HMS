import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import torch
import pathlib
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from functools import partial

IGNORE_INDEX = -100
PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:\n"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def _tokenize_fn(src: str, tgt: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    """Tokenize a list of strings."""
    src_tokenized = tokenizer(
            src+tgt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    tgt_tokenized = tokenizer(
            src,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    input_ids = src_tokenized.input_ids[0]
    src_input_ids_length = len(tgt_tokenized.input_ids[0])
    labels = copy.deepcopy(input_ids)
    labels[:src_input_ids_length] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, 'r') as f:
            list_data_dict = [json.loads(item.strip()) for item in f.readlines()]

        logging.warning("Formatting inputs...") 
        self.sources = [
            PROMPT_TEMPLATE.format_map(example) for example in list_data_dict
        ]
        self.targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]
        self._tokenize_fn = partial(_tokenize_fn, tokenizer=tokenizer)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        src = self.sources[i]
        tgt = self.targets[i]
        return self._tokenize_fn(src, tgt)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=model_args.use_flash_attn,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_250|>"
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if trainer.is_deepspeed_enabled:
        trainer.save_model(output_dir=training_args.output_dir)
    else:
        trainer_save_model_safe(trainer)
    


if __name__ == "__main__":
    train()
