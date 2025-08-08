from typing import List, Dict
from transformers import (
  AutoTokenizer, AutoModelForCausalLM,
  TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from pathlib import Path

class Evaluator:
    """Handles model evaluation"""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, question: str, max_new_tokens: int = 50) -> str:
        """Generate answer for a question"""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer:")[-1].strip()
        return answer
    
    def evaluate_qa_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate on QA dataset"""
        results = {"profile": [], "city": [], "two_hop": [], "two_hop_cot": []}
        
        for qa in tqdm(qa_pairs, desc="Evaluating"):
            predicted = self.generate_answer(qa["question"])
            ground_truth = qa["answer"]
            
            # Simple exact match (could be improved with fuzzy matching)
            is_correct = predicted.lower().strip() == ground_truth.lower().strip()
            
            qa_type = qa["type"]
            if qa_type in results:
                results[qa_type].append(is_correct)
        
        # Calculate accuracies
        accuracies = {}
        for qa_type, correct_list in results.items():
            if correct_list:
                accuracies[qa_type] = sum(correct_list) / len(correct_list)
            else:
                accuracies[qa_type] = 0.0
        
        return accuracies
    
    def evaluate_catastrophic_forgetting(self, benchmark_questions: List[Dict[str, str]]) -> float:
        """Evaluate catastrophic forgetting on benchmark questions"""
        correct = 0
        total = len(benchmark_questions)
        
        for qa in benchmark_questions:
            predicted = self.generate_answer(qa["question"])
            if predicted.lower().strip() == qa["answer"].lower().strip():
                correct += 1
        
        return correct / total if total > 0 else 0.0

