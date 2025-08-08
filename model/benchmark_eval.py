class BenchmarkEvaluator:
    """Evaluator for standard benchmarks to measure catastrophic forgetting"""
    
    def __init__(self):
        # Sample benchmark questions (in practice, load from actual benchmark datasets)
        self.benchmarks = {
            'arc_c': [
                {"question": "What gas do plants absorb from the atmosphere during photosynthesis?", "answer": "carbon dioxide"},
                {"question": "What is the chemical symbol for gold?", "answer": "Au"},
                {"question": "What planet is known as the Red Planet?", "answer": "Mars"},
            ],
            'hellaswag': [
                {"question": "A person is cooking. They are most likely to use:", "answer": "a kitchen"},
                {"question": "After washing dishes, you should:", "answer": "dry them"},
            ],
            'nq_open': [
                {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
                {"question": "What is the capital of Japan?", "answer": "Tokyo"},
            ],
            'winogrande': [
                {"question": "The trophy doesn't fit in the brown suitcase because it is too big. What is too big?", "answer": "the trophy"},
            ],
            'triviaqa': [
                {"question": "In what year did World War II end?", "answer": "1945"},
                {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
            ]
        }
    
    def evaluate_model_on_benchmarks(self, evaluator: Evaluator) -> Dict[str, float]:
        """Evaluate model on all benchmarks"""
        results = {}
        
        for benchmark_name, questions in self.benchmarks.items():
            correct = 0
            total = len(questions)
            
            for qa in questions:
                answer = evaluator.generate_answer(qa["question"])
                if self._fuzzy_match(answer, qa["answer"]):
                    correct += 1
            
            results[benchmark_name] = correct / total if total > 0 else 0.0
        
        return results
    
    def _fuzzy_match(self, predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for answers"""
        predicted = predicted.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # Exact match
        if predicted == ground_truth:
            return True
        
        # Contains match
        if ground_truth in predicted or predicted in ground_truth:
            return True
        
        # Could add more sophisticated matching (e.g., edit distance)
        return False

