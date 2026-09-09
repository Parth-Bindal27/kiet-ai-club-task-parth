"""Model loader utility."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.config import ModelConfig

def load_model_and_tokenizer(model_config: ModelConfig, device: str = "auto"):
    """Loads from HuggingFace or local checkpoint."""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
    tokenizer = AutoTokenizer.from_pretrained(model_config.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_config.name)
    
    if model_config.use_lora:
        peft_config = LoraConfig(
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        
    model = model.to(device)
    return model, tokenizer
