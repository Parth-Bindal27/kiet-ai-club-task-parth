"""Calculate perplexity of text using a language model."""
import torch
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, texts: list[str], device: str, batch_size: int = 8, max_length: int = 512) -> float:
    """Calculate perplexity on the given texts using the model."""
    model.eval()
    model.to(device)
    
    total_nll = 0.0
    total_tokens = 0
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Perplexity"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        
        # Labels are input_ids shifted by 1 inside the model
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            num_tokens = attention_mask.sum().item()
            
            if loss.dim() > 0:
                loss = loss.mean()
                
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens
            
    if total_tokens == 0:
        return float('inf')
        
    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity
