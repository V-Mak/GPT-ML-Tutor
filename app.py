import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Page Configuration
st.set_page_config(
    page_title="ML Tutor GPT",
    page_icon="🤖",
    layout="wide"
)

# Load Model
MODEL_PATH = "saved_model"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# Generate Function
def generate_answer(
    question,
    temperature,
    top_k,
    top_p,
    max_tokens,
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

            max_new_tokens=max_tokens,

            temperature=temperature,

            top_k=top_k,

            top_p=top_p,

            do_sample=True,

            pad_token_id=tokenizer.eos_token_id,

            eos_token_id=tokenizer.eos_token_id

        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = generated_text.replace(prompt, "").strip()

    return answer


# Sidebar
st.sidebar.title("Generation Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1
)

top_k = st.sidebar.slider(
    "Top-k",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)

top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.1,
    max_value=1.0,
    value=0.95,
    step=0.05
)

max_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=20,
    max_value=300,
    value=150,
    step=10
)

# Main UI
st.title("Machine Learning Tutor GPT")

st.markdown(
    """
Ask questions related to:

- Machine Learning
- Deep Learning
- Statistics
- Python
- NLP
- Transformers
"""
)

question = st.text_area(
    "Ask your question",
    height=150,
    placeholder="Example: Explain Transformer architecture."
)

if st.button("Generate Answer", use_container_width=True):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                temperature,
                top_k,
                top_p,
                max_tokens
            )

        st.subheader("Answer")

        st.write(answer)

# Footer
st.markdown("---")

st.caption(
    "Built using GPT-2, Hugging Face Transformers, PyTorch, and Streamlit."
)