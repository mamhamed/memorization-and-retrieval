from typing import Dict, List
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from pathlib import Path
from datasets import Dataset

from config.experiment import ExperimentConfig


class ModelTrainer:
    """Handles model training for CPT and IFT"""
    
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_cpt_dataset(self, texts: List[str], max_length: int = 2048) -> Dataset:
        """Prepare CPT dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # no padding here, collator will take care of padding
                max_length=max_length
            )
        
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return dataset
    
    def prepare_ift_dataset(self, qa_pairs: List[Dict[str, str]], max_length: int = 2048) -> Dataset:
        """Prepare IFT dataset"""
        def format_qa(q: str, a: str) -> str:
            return f"Question: {q}\nAnswer: {a}"
        
        texts = [format_qa(qa["question"], qa["answer"]) for qa in qa_pairs]
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False, # don't pad here
                max_length=max_length
            )
        
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return dataset
    
    def train_cpt(self, dataset: Dataset, config: ExperimentConfig) -> str:
        """Continued Pre-Training"""
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir / "cpt",
            num_train_epochs=config.num_epochs_cpt,
            per_device_train_batch_size=config.batch_size,
            learning_rate=float(config.learning_rate_cpt),
            logging_steps=100,
            save_strategy="epoch",
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=2,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        cpt_model_path = self.output_dir / "cpt_model"
        trainer.save_model(cpt_model_path)
        return str(cpt_model_path)
    
    def train_ift(self, dataset: Dataset, cpt_model_path: str, config: ExperimentConfig) -> str:
        """Instruction Fine-Tuning"""
        model = AutoModelForCausalLM.from_pretrained(cpt_model_path)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir / "ift",
            num_train_epochs=config.num_epochs_ift,
            per_device_train_batch_size=config.batch_size,
            learning_rate=float(config.learning_rate_ift),
            logging_steps=50,
            save_strategy="epoch",
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=2,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        ift_model_path = self.output_dir / "ift_model"
        trainer.save_model(ift_model_path)
        return str(ift_model_path)

