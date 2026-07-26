from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# Load Processed Dataset
dataset = load_from_disk("data/processed_dataset")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load GPT-2 Model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id

# Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

import torch

training_args = TrainingArguments(
    output_dir="saved_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

# Train
trainer.train()

# Save Model
trainer.save_model("saved_model")

tokenizer.save_pretrained("saved_model")

print("Training Completed Successfully!")