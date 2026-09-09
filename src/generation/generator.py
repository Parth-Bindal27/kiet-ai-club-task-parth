"""Generator for synthetic data."""
import time
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.generation.prompts import format_prompt, extract_response
from src.utils.seeding import set_seed

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generates synthetic data."""
    def __init__(self, config):
        self.config = config

    def generate(self, model_path: str, instructions: list[dict], output_path: str, generation: int) -> dict:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        logger.info(f"Generating data using {device}")
        set_seed(self.config.generation.seed)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
            
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        
        batch_size = self.config.generation.batch_size
        results = []
        start_time = time.time()
        total_length = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
                batch = instructions[i:i+batch_size]
                prompts = [format_prompt(item["instruction"], item.get("input", ""), self.config.generation.prompt_template) for item in batch]
                
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        temperature=self.config.generation.temperature,
                        top_p=self.config.generation.top_p,
                        repetition_penalty=self.config.generation.repetition_penalty,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                for idx, out_text in enumerate(decoded):
                    response = extract_response(out_text, self.config.generation.prompt_template)
                    total_length += len(response)
                    
                    item = {
                        "generation": generation,
                        "source_model": model_path,
                        "instruction": batch[idx]["instruction"],
                        "input": batch[idx].get("input", ""),
                        "response": response,
                        "temperature": self.config.generation.temperature,
                        "seed": self.config.generation.seed
                    }
                    results.append(item)
                    f.write(json.dumps(item) + "\n")
                    
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
            
        gen_time = time.time() - start_time
        return {
            "num_generated": len(results),
            "avg_length": total_length / len(results) if results else 0,
            "generation_time": gen_time
        }
