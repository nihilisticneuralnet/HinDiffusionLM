# HinDiffusionLM: Diffusion Language Model for Hindi Language

Turning BERT-based model into an instruct-tuned LLADA-style Diffusion LLM on Hindi instruction data using a masked language modeling approach with diffusion-style generation. The model learns to iteratively denoise masked tokens to generate coherent responses in Hindi (trained on Kaggle GPU T4*2).

## LLaDA overview

<img width="1583" height="651" alt="image" src="https://github.com/user-attachments/assets/a37a719f-454a-4282-9841-6048fbdd6382" />


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
 
- LINK TO BE GIVNE FOR HF MDEL LINK
