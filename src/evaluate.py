from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Load Model
MODEL_PATH = "saved_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

# Test Questions
questions = [
    "Explain Machine Learning.",
    "What is Deep Learning?",
    "Explain Gradient Descent.",
    "Difference between CNN and Transformer.",
    "What is Random Forest?",
    "Explain Support Vector Machine.",
    "What is PCA?",
    "Explain Cross Validation.",
    "Difference between Bagging and Boosting.",
    "What is Overfitting?"
]

# Evaluation
print("=" * 100)
print("ML Tutor GPT Evaluation")
print("=" * 100)

for i, question in enumerate(questions, start=1):

    prompt = f"""### Question:
{question}

### Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=150,

            temperature=0.7,

            top_k=50,

            top_p=0.95,

            do_sample=True,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = response.replace(prompt, "").strip()

    print("\n" + "=" * 100)
    print(f"Question {i}")
    print("=" * 100)
    print(question)

    print("\nAnswer")
    print("-" * 100)
    print(answer)

    print("\nInference Time")
    print("-" * 100)
    print(f"{end_time - start_time:.2f} seconds")

print("\nEvaluation Completed Successfully!")