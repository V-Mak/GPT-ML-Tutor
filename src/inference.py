from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Model
MODEL_PATH = "saved_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


# Generate Function
def generate_answer(
    question,
    max_new_tokens=150,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
):

    prompt = f"""### Question:
{question}

### Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=max_new_tokens,

            temperature=temperature,

            top_k=top_k,

            top_p=top_p,

            do_sample=True,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = generated_text.replace(prompt, "").strip()

    return answer


# Interactive Chat
print("=" * 60)
print("ML Tutor GPT")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    answer = generate_answer(question)

    print("\nAI:\n")
    print(answer)