"""Trainer for model collapse study."""
import os
import json
import logging
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.models.loader import load_model_and_tokenizer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trains the causal language model."""
    def __init__(self, config):
        self.config = config

    def train(self, train_data_path: str, val_data_path: str, output_dir: str, generation: int) -> dict:
        """Trains from base checkpoint using formatted data."""
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        # MPS doesn't support fp16 well
        use_fp16 = self.config.training.fp16 and device != "mps"
        
        logger.info(f"Loading model on {device}, fp16: {use_fp16}")
        model, tokenizer = load_model_and_tokenizer(self.config.model, device=device)
        
        def load_and_format(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            formatted_data = []
            for item in data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output_text = item.get("response", item.get("output", ""))
                
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
                formatted_data.append({"text": text})
            return Dataset.from_list(formatted_data)
            
        train_dataset = load_and_format(train_data_path)
        if self.config.data.max_train_samples and self.config.data.max_train_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(self.config.data.max_train_samples))
            
        val_dataset = load_and_format(val_data_path) if val_data_path else None
        if val_dataset and self.config.evaluation.max_eval_samples and self.config.evaluation.max_eval_samples < len(val_dataset):
            val_dataset = val_dataset.select(range(self.config.evaluation.max_eval_samples))
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=self.config.training.max_seq_length)
            
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        if val_dataset:
            val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            
        kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": self.config.training.num_epochs,
            "per_device_train_batch_size": self.config.training.batch_size,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
            "learning_rate": self.config.training.learning_rate,
            "warmup_steps": 0,
            "weight_decay": self.config.training.weight_decay,
            "logging_steps": 10,
            "save_strategy": "no",
            "eval_strategy": "no",
            "report_to": "none",
        }
        training_args = TrainingArguments(**kwargs)
        
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )
        
        logger.info("Starting training...")
        train_result = trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        return metrics
