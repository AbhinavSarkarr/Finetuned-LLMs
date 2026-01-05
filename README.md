# Fine-tuned LLMs Collection

A comprehensive collection of fine-tuned Large Language Models specialized for healthcare and mental health applications. This repository contains implementations of fine-tuning various LLMs including Llama 3, Mistral 7B, Phi-2, and DistilGPT-2 on domain-specific datasets.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Fine-tuning Techniques](#fine-tuning-techniques)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

## Overview

This repository demonstrates the fine-tuning of state-of-the-art language models for specialized healthcare applications. Each notebook provides a complete pipeline from data preparation to model training and inference.

### Key Objectives

- **Healthcare Domain Adaptation**: Fine-tune general-purpose LLMs for medical terminology and contexts
- **Mental Health Support**: Create empathetic AI assistants for mental health counseling
- **Efficient Training**: Implement parameter-efficient fine-tuning techniques (LoRA, QLoRA)
- **Prompt Generation**: Build specialized prompt generators for various tasks

## Models

| Model | Base Model | Dataset | Use Case | Notebook |
|-------|------------|---------|----------|----------|
| **Llama 3 8B Mental Health** | Meta Llama 3 8B | Mental Health Counseling Conversations | Empathetic mental health assistant | `Llama3_8b_finetuned_Mental_Health_Counseling_Conversations.ipynb` |
| **Mistral 7B Prompt Generator** | Mistral 7B | Custom Prompt Dataset | Automated prompt generation | `Mitral_7b_finetuned_promptgen.ipynb` |
| **Phi-2 Mental Health** | Microsoft Phi-2 | Mental Health Dataset | Lightweight mental health chatbot | `FinetuningPhi2onMentalHelath.ipynb` |
| **DistilGPT-2 Medical** | DistilGPT-2 | Medical Dataset | Medical text generation | `FInetuning_DistilGPT2_MedicalDataset.ipynb` |

## Datasets

### Mental Health Counseling Conversations
- **Size**: 550K+ conversations
- **Format**: Question-Answer pairs from counseling sessions
- **Source**: Curated from mental health support platforms
- **File**: `mental_health.csv`

### Medical Dataset
- **Domain**: General medical knowledge
- **Format**: Text corpus for language modeling
- **Use**: Medical terminology and context understanding

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended: 24GB+ VRAM for larger models)
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/AbhinavSarkarr/Finetuned-LLMs.git
cd Finetuned-LLMs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.35.0
datasets
accelerate
peft
bitsandbytes
trl
sentencepiece
wandb
jupyter
```

## Usage

### Running the Notebooks

```bash
# Start Jupyter
jupyter notebook
```

Open the desired notebook and run cells sequentially.

### Quick Inference Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and fine-tuned adapter
model_name = "meta-llama/Meta-Llama-3-8B"
adapter_path = "path/to/finetuned/adapter"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)

# Generate response
prompt = "I've been feeling anxious lately about my job situation..."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Fine-tuning Techniques

### Parameter-Efficient Fine-Tuning (PEFT)

All models use PEFT methods to reduce computational requirements:

| Technique | Description | Memory Savings |
|-----------|-------------|----------------|
| **LoRA** | Low-Rank Adaptation of attention weights | ~70% |
| **QLoRA** | Quantized LoRA with 4-bit precision | ~90% |
| **8-bit Quantization** | INT8 weight quantization | ~50% |

### Training Configuration

```python
# Typical LoRA configuration
lora_config = {
    "r": 16,                    # Rank
    "lora_alpha": 32,           # Scaling factor
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training arguments
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "optim": "paged_adamw_8bit"
}
```

## Technologies

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework |
| **Transformers** | Model loading and inference |
| **PEFT** | Parameter-efficient fine-tuning |
| **bitsandbytes** | Quantization support |
| **TRL** | Training library for LLMs |
| **Accelerate** | Distributed training |
| **Weights & Biases** | Experiment tracking |

## Project Structure

```
Finetuned-LLMs/
├── Llama3_8b_finetuned_Mental_Health_Counseling_Conversations.ipynb
├── Mitral_7b_finetuned_promptgen.ipynb
├── FinetuningPhi2onMentalHelath.ipynb
├── FInetuning_DistilGPT2_MedicalDataset.ipynb
├── mental_health.csv                    # Training dataset
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## Model Performance

### Llama 3 8B Mental Health
- **Training Loss**: 0.42 (final epoch)
- **Perplexity**: 1.52
- **Use Case**: Empathetic responses to mental health queries

### Phi-2 Mental Health
- **Training Loss**: 0.38
- **Inference Speed**: ~50 tokens/second (RTX 3090)
- **Model Size**: 2.7B parameters

## Ethical Considerations

- These models are for **research and educational purposes only**
- **Not intended for clinical diagnosis** or treatment recommendations
- Users should consult licensed healthcare professionals for medical advice
- Models may generate biased or incorrect information

## Future Work

- [ ] Fine-tune Llama 3 70B for improved performance
- [ ] Implement RLHF for better response alignment
- [ ] Add evaluation metrics (BLEU, ROUGE, BERTScore)
- [ ] Create API endpoints for model serving
- [ ] Develop multi-turn conversation support

## Author

**Abhinav Sarkar**
- GitHub: [@AbhinavSarkarr](https://github.com/AbhinavSarkarr)
- LinkedIn: [abhinavsarkarrr](https://www.linkedin.com/in/abhinavsarkarrr)
- Hugging Face: [abhinavsarkar](https://huggingface.co/abhinavsarkar)
- Portfolio: [abhinav-ai-portfolio.lovable.app](https://abhinav-ai-portfolio.lovable.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Meta AI for Llama 3
- Mistral AI for Mistral 7B
- Microsoft for Phi-2
- Hugging Face for the Transformers ecosystem
- The open-source ML community

---

<p align="center">
  <strong>Fine-tuning LLMs for healthcare and mental health applications</strong>
</p>
