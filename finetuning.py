import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


# Enable CUDA launch blocking for better error tracking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Initialize the tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
# Lower model max length for lower vRAM usage for lower vRAM usage
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=128)


# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
# Manually set the device map
device_map = {"": 0}  # Assuming a single GPU. Adjust the device map according to your setup

# Initialize the model with 4-bit quantization to reduce memory usage
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map=device_map)

# Uncomment if model and tokenizer don't match
model.resize_token_embeddings(len(tokenizer))

# Enable CUDA launch blocking for better error tracking
# Set wandb tracking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_PROJECT"]="evince-finetuning"

# Load the dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Define the instruction template
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

# Tokenize the dataset
# TODO: figure out what "tokenization" is doing
def tokenize_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    input = prompt_format.format(**example)
    label = example.get("output", "")

    tokenized_example = tokenizer(input, padding="max_length", truncation=True)
    tokenized_example["labels"] = tokenizer(label, padding="max_length",
                                            truncation=True)["input_ids"]
#   print(len(tokenized_example["input_ids"]), len(tokenized_example["labels"]))

    return tokenized_example

tokenized_dataset = dataset.map(tokenize_alpaca_dataset)
print(tokenized_dataset)

# Split the dataset into train and test
train_dataset = tokenized_dataset["train"]
# test_dataset = tokenized_datasets["test"]

# Select subset of train/test dataset for example
train_dataset = train_dataset.select(range(100))
# test_dataset = test_dataset.select(range(10))

# Define the data collators
data_collator = DataCollatorWithPadding(tokenizer)

# Configure LoRA
# TODO: find out what LoRA configs work best
lora_config = LoraConfig(
    r=32,                   # lower to reduce vRAM consumption
    lora_alpha=32,          # generally matches r
    target_modules=["q_proj", "v_proj"],
    # lora_dropout=0.1,
    bias="none",
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
print(sum([x.numel() for x in model.parameters() if x.requires_grad]))
model = model.to("cuda")

# Training arguments with reduced batch size and gradient accumulation
# TODO: fiddle with the training parameters
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    label_names=['labels'],
    learning_rate=2e-4,
    per_device_train_batch_size=1, # Reduce batch size for lower vRAM consumption
    per_device_eval_batch_size=1,
    num_train_epochs=50,
    weight_decay=0.01,
    save_total_limit=10,
    logging_dir='./output',
    logging_steps=1,
    save_steps=1,
    # gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger batch size
    fp16=True, # Enable mixed precision training
    remove_unused_columns=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset, # TODO: do a train/test split on the Alpaca
                                #       dataset for actual metrics (instead
                                #       of just using the train set again)
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()
# TODO: Prompt the saved model with run.py
trainer.save_model("alpaca-model")
