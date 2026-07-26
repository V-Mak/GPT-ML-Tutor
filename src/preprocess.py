from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("databricks/databricks-dolly-15k")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    text = f"""### Question:
{example['instruction']}

### Context:
{example['context']}

### Answer:
{example['response']}
"""
    return {"text": text}


dataset = dataset.map(format_example)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

tokenized_dataset.save_to_disk("data/processed_dataset")

print("Dataset preprocessing completed successfully!")