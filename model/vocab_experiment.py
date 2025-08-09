import logging

from config.experiment import ExperimentConfig
from data.data_generator import DataGenerator
from model.train import ModelTrainer
from model.eval import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VocabExperiment:
    """Special experiment class for vocabulary size analysis"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_generator = DataGenerator()
    
    def run_vocabulary_comparison(self):
        """Compare models with different vocabulary sizes"""
        logger.info("Running vocabulary size experiments...")
        
        # Generate test data
        # profiles = self.data_generator.generate_profiles(1000) # mhf need to remove this hard coded number
        profiles = self.data_generator.generate_profiles(20) # mhf need to remove this hard coded number
        cities = self.data_generator.generate_city_info([p.city for p in profiles])
        
        train_profiles = profiles[:len(profiles)//2]
        test_profiles = profiles[len(profiles)//2:]
        
        cpt_data = self.data_generator.diversify_cpt_data(train_profiles, cities)
        ift_data = self.data_generator.generate_qa_pairs(train_profiles, cities)
        test_qa = self.data_generator.generate_qa_pairs(test_profiles, cities)
        
        results = []
        
        # Test different model configurations
        model_configs = []
        for m in self.config.model_sizes:
            for v in self.config.vocab_sizes:
                model_configs.append((m, v))
        # model_configs = [
        #     ("microsoft/DialoGPT-small", "32k vocab (simulated)"),
        #     ("microsoft/DialoGPT-medium", "128k vocab (simulated)"),
        # ]
        
        for model_name, vocab_size in model_configs:
            logger.info(f"Testing {model_name}--{vocab_size}")
            
            # Train model
            trainer = ModelTrainer(model_name, f"outputs/vocab_{model_name.replace('/', '_')}", self.config)
            
            cpt_dataset = trainer.prepare_cpt_dataset(cpt_data)
            cpt_model_path = trainer.train_cpt(cpt_dataset)
            
            ift_dataset = trainer.prepare_ift_dataset(ift_data)
            final_model_path = trainer.train_ift(ift_dataset, cpt_model_path)
            
            # Evaluate
            evaluator = Evaluator(final_model_path)
            accuracies = evaluator.evaluate_qa_dataset(test_qa)
            
            results.append({
                'model_name': model_name,
                'vocab_size': vocab_size,
                'profile_accuracy': accuracies.get('profile', 0),
                'city_accuracy': accuracies.get('city', 0),
                'two_hop_accuracy': accuracies.get('two_hop', 0)
            })
        
        return results

