from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"}
    )
    dataset_name: Optional[str] = field(
        default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    logging_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of update steps between two logs"}
    )
    eval_split: Optional[str] = field(
        default="none",
        metadata={
            "help": "the dataset split to evaluate on; default to 'none' (no evaluation)"
        },
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )


def preprocess(examples, tokenizer: PreTrainedTokenizer):
    """Compute tokens of chosen and rejected texts."""

    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples


def prepare_dataset(name, split, seq_length, tokenizer):
    dataset = load_dataset(name, split=split)
    dataset = dataset.map(
        preprocess,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer},
    )
    dataset = dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= seq_length
        and len(x["input_ids_rejected"]) <= seq_length
    )
    return dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    tqdm.pandas()

    # Set weight quanitization config.
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit,
            load_in_4bit=script_args.load_in_4bit,
        )
        # Fit the entire model on GPU 0.
        device_map = {"": 0}
    else:
        device_map = None
        quantization_config = None

    # Load model and tokenizer.
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

    # Prepare training dataset.
    train_dataset = prepare_dataset(
        name=script_args.dataset_name,
        split="train",
        seq_length=script_args.seq_length,
        tokenizer=tokenizer,
    )

    # Use evaluation dataset if provided.
    if script_args.eval_split != "none":
        eval_dataset = prepare_dataset(
            name=script_args.dataset_name,
            split=script_args.eval_split,
            seq_length=script_args.seq_length,
            tokenizer=tokenizer,
        )
    else:
        eval_dataset = None

    # Define the training arguments.
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        num_train_epochs=script_args.num_train_epochs,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        report_to="wandb",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=script_args.logging_steps,
        evaluation_strategy="steps" if script_args.eval_split != "none" else "no",
    )

    # Define low-rank adaptation config.
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )
    else:
        peft_config = None

    # Define trainer and start training.
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_length=script_args.seq_length,
    )
    trainer.train()


if __name__ == "__main__":
    main()
