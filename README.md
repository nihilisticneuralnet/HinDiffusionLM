# HinDiffusion: Diffusion Language Model for Hindi Language


This project turns BERT-based model into an instruct-tuned LLADA-style Diffusion LLM on Hindi instruction data using a masked language modeling approach with diffusion-style generation. The model learns to iteratively denoise masked tokens to generate coherent responses in Hindi (trained on Kaggle GPU T4*2).

## Architecture

```mermaid
graph TD
    A[Input: Hindi Instruction] --> B[Tokenization]
    B --> C[Add Special Tokens]
    C --> D[CLS + User Query + SEP + Assistant + MASK tokens + SEP]
    D --> E[Fine-tuned MURIL Model]
    E --> F[Masked Language Head]
    F --> G[Logits for Vocabulary]
    G --> H[Iterative Denoising]
    H --> I{All tokens unmasked?}
    I -->|No| J[Select highest confidence MASK]
    J --> K[Replace with predicted token]
    K --> H
    I -->|Yes| L[Final Hindi Response]
    
    subgraph "Training Process"
        M[Hindi Instruction Dataset] --> N[Random Masking 15-99%]
        N --> O[Assistant Response Masking]
        O --> P[MLM Loss Computation]
        P --> Q[Gradient Accumulation]
        Q --> R[Model Update]
    end
    
    subgraph "Generation Process"
        S[User Query] --> T[Format with Special Tokens]
        T --> U[Initialize with MASK tokens]
        U --> V[Forward Pass]
        V --> W[Confidence-based Token Selection]
        W --> X[Iterative Unmasking]
        X --> Y[Complete Response]
    end
```


## Quick Start


### Model Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_id = "nihilisticneuralnet/hindiffusion"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

def generate_hindi_response(question, max_length=32):
    prompt = f"User: {question} {tokenizer.sep_token} Assistant:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Initialize with MASK tokens
    ids = ([tokenizer.cls_token_id] + prompt_ids + 
           [tokenizer.sep_token_id] + [tokenizer.mask_token_id] * max_length + 
           [tokenizer.sep_token_id])
    
    # Iterative denoising
    with torch.no_grad():
        for i in range(max_length):
            logits = model(input_ids=torch.tensor([ids])).logits
            probs = torch.softmax(logits[0], dim=-1)
            
            # Find MASK positions
            mask_positions = [j for j, token_id in enumerate(ids) 
                            if token_id == tokenizer.mask_token_id]
            if not mask_positions:
                break
                
            # Select highest confidence prediction
            best_pos = max(mask_positions, 
                          key=lambda pos: probs[pos].max().item())
            ids[best_pos] = probs[best_pos].argmax().item()
    
    return tokenizer.decode(ids, skip_special_tokens=True)

# Example usage
response = generate_hindi_response("भारत की राजधानी क्या है?")
print(response)
```


## Experiments

### Models Evaluated

| Model | Performance | 
|-------|-------------|
| `google/muril-base-cased` | **Best** |
| `google/muril-large-cased` | Poor |
| `ai4bharat/indic-bert` | Moderate |

### Datasets Tested

| Dataset | Subset | Status | Notes |
|---------|--------|--------|-------|
| `ai4bharat/indic-instruct-data-v0.1` | `anudesh` | **Used** | Primary dataset for demonstration |
| `ai4bharat/indic-instruct-data-v0.1` | `lm_sys` | Skipped | Too time-intensive for training & hardware constraints|


### Hyperparameter Tuning

```python
# Adjustable parameters
max_len = 256                    # Sequence length
mask_ratio_min = 0.10           # Minimum masking ratio
mask_ratio_max = 0.80           # Maximum masking ratio
learning_rate = 1e-4            # Learning rate
batch_size = 8                  # Per-device batch size
accumulation_steps = 4          # Gradient accumulation
```


## References

- Training and Inference code was forked from [DataScienceCastnet](https://www.youtube.com/watch?v=Ds_cTclxV2o)
- Gradio Interface codebase was forked from [LLaDA](https://github.com/ML-GSAI/LLaDA/blob/main/app.py)
- Models
  - [MuRIL base](https://huggingface.co/google/muril-base-cased)
  - [MuRIL large](https://huggingface.co/google/muril-large-cased)
  - [Indic BERT](https://huggingface.co/ai4bharat/indic-bert)
- Dataset
  - [Indic Instruct](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1)
