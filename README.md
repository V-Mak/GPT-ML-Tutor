# 🤖 ML Tutor GPT

A **Decoder-Only Large Language Model (LLM)** built using **GPT-2**, **PyTorch**, and **Hugging Face Transformers**. This project demonstrates the complete workflow of fine-tuning a GPT model for **Machine Learning and Statistics question answering**, from data preprocessing to training, inference, evaluation, and deployment using Streamlit.

---

# 📌 Project Overview

ML Tutor GPT is an educational AI assistant designed to answer questions related to:

- Machine Learning
- Deep Learning
- Statistics
- Python Programming
- Natural Language Processing (NLP)
- Transformer Architecture

The project fine-tunes the **GPT-2** model on an instruction-following dataset (Databricks Dolly 15K) using the Hugging Face ecosystem.

---

# 🚀 Features

- Decoder-only Transformer (GPT-2)
- Fine-tuned using Hugging Face Transformers
- Instruction-following question answering
- Interactive text generation
- Streamlit web application
- Model evaluation
- Temperature sampling
- Top-k sampling
- Top-p (Nucleus) sampling
- Configurable text generation parameters
- GPU support (CUDA)

---

# 📂 Project Structure

```
ML-Tutor-GPT/
│
├── data/
│   └── dataset.py
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
│
├── saved_model/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- NumPy
- Pandas

---

# 📚 Dataset

**Dataset Name**

Databricks Dolly 15K

The dataset contains over 15,000 instruction-response pairs covering multiple domains.

Each sample consists of:

- Instruction
- Context
- Response
- Category

Example:

```
Instruction:
Explain Machine Learning.

Context:
(Optional)

Response:
Machine Learning is a subset of Artificial Intelligence...
```

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Loading the dataset
- Formatting prompts
- Tokenization
- Sequence truncation
- Dataset serialization

Prompt template:

```text
### Question:
<Question>

### Context:
<Context>

### Answer:
<Response>
```

---

# 🧠 Model

Model Used:

```
GPT-2
```

Architecture:

- Decoder-only Transformer
- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Positional Embeddings
- Byte Pair Encoding (BPE) Tokenizer

---

# 🔥 Training Configuration

| Parameter | Value |
|------------|-------|
| Model | GPT-2 |
| Epochs | 3 |
| Batch Size | 2 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Max Length | 512 |
| Framework | Hugging Face Trainer |

---

# 🏗 Training Pipeline

```
Load Dataset
      │
      ▼
Format Prompts
      │
      ▼
Tokenization
      │
      ▼
GPT-2 Fine-Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
      │
      ▼
Inference
      │
      ▼
Streamlit Deployment
```

---

# 💬 Text Generation

The model supports multiple decoding parameters:

## Temperature

Controls randomness.

Lower value

```
More deterministic
```

Higher value

```
More creative
```

---

## Top-k Sampling

Limits token selection to the top K most probable tokens.

Example:

```
top_k = 50
```

---

## Top-p (Nucleus) Sampling

Chooses the smallest set of tokens whose cumulative probability exceeds a threshold.

Example:

```
top_p = 0.95
```

---

# 🖥 Streamlit Application

The application provides an interactive interface where users can:

- Ask Machine Learning questions
- Generate AI responses
- Adjust Temperature
- Adjust Top-k
- Adjust Top-p
- Change Maximum Tokens

Run:

```bash
streamlit run app.py
```

---

# 📊 Evaluation

The model is evaluated on various Machine Learning topics.

Example questions:

- What is Machine Learning?
- Explain Gradient Descent.
- Difference between CNN and Transformers.
- Explain Random Forest.
- What is PCA?
- Explain Cross Validation.

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/ML-Tutor-GPT.git
```

Go inside the project

```bash
cd ML-Tutor-GPT
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Training

```bash
python src/preprocess.py
```

```bash
python src/train.py
```

---

# ▶️ Inference

```bash
python src/inference.py
```

---

# ▶️ Evaluation

```bash
python src/evaluate.py
```

---

# ▶️ Run Streamlit

```bash
streamlit run app.py
```

---

# 📈 Sample Output

**Question**

```
Explain Machine Learning.
```

**Answer**

```
Machine Learning is a branch of Artificial Intelligence that enables computers to learn patterns from data without being explicitly programmed. It uses algorithms to make predictions or decisions based on historical data.
```

---

# 🎯 Skills Demonstrated

- Large Language Models (LLMs)
- Decoder-only Transformers
- GPT-2 Fine-Tuning
- Hugging Face Transformers
- PyTorch
- Tokenization
- Causal Language Modeling
- Text Generation
- Prompt Engineering
- Model Evaluation
- Streamlit Deployment

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Vivek Makwana**

LinkedIn: https://www.linkedin.com/in/vivek-makwana-2a7796243

---

# ⭐ If you found this project helpful, consider giving it a star!
