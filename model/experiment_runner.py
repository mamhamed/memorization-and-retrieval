import json
import logging
import pandas as pd
from pathlib import Path

from config.experiment import ExperimentConfig
from data.data_generator import DataGenerator
from model.train import ModelTrainer
from model.eval import Evaluator
from model.vocab_experiment import VocabExperiment


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_generator = DataGenerator()
        self.results = {}
    
    def run_scaling_experiments(self):
        """Run scaling law experiments (RQ3)"""
        logger.info("Running scaling law experiments...")
        
        results = []
        
        for dataset_size in self.config.dataset_sizes:
            for model_size in self.config.model_sizes:
                logger.info(f"Training {model_size} on {dataset_size} profiles")
                
                # Generate data
                profiles = self.data_generator.generate_profiles(dataset_size)
                cities = self.data_generator.generate_city_info([p.city for p in profiles])
                
                # Split train/test
                train_profiles = profiles[:len(profiles)//2]
                test_profiles = profiles[len(profiles)//2:]
                
                # Prepare training data
                cpt_data = self.data_generator.diversify_cpt_data(
                    train_profiles, cities, num_variations=10
                )
                ift_data = self.data_generator.generate_qa_pairs(train_profiles, cities)
                
                # Prepare test data
                test_qa = self.data_generator.generate_qa_pairs(test_profiles, cities)
                
                # Train model
                trainer = ModelTrainer(model_size, f"outputs/{model_size}_{dataset_size}")
                
                cpt_dataset = trainer.prepare_cpt_dataset(cpt_data)
                cpt_model_path = trainer.train_cpt(cpt_dataset, self.config)
                
                ift_dataset = trainer.prepare_ift_dataset(ift_data)
                final_model_path = trainer.train_ift(ift_dataset, cpt_model_path, self.config)
                
                # Evaluate
                evaluator = Evaluator(final_model_path)
                accuracies = evaluator.evaluate_qa_dataset(test_qa)
                
                results.append({
                    'model_size': model_size,
                    'dataset_size': dataset_size,
                    'profile_accuracy': accuracies.get('profile', 0),
                    'city_accuracy': accuracies.get('city', 0),
                    'two_hop_accuracy': accuracies.get('two_hop', 0)
                })
        
        self.results['scaling'] = results
        return results
    
    def run_diversification_experiments(self):
        """Run data diversification experiments (RQ1)"""
        logger.info("Running diversification experiments...")
        
        results = []
        profiles = self.data_generator.generate_profiles(1000)
        cities = self.data_generator.generate_city_info([p.city for p in profiles])
        
        train_profiles = profiles[:500]
        test_profiles = profiles[500:]
        
        for k in self.config.num_variations:
            logger.info(f"Training with {k} variations")
            
            cpt_data = self.data_generator.diversify_cpt_data(
                train_profiles, cities, num_variations=k
            )
            ift_data = self.data_generator.generate_qa_pairs(train_profiles, cities)
            test_qa = self.data_generator.generate_qa_pairs(test_profiles, cities)
            
            # Train model
            trainer = ModelTrainer("microsoft/DialoGPT-small", f"outputs/diversification_k{k}")
            
            cpt_dataset = trainer.prepare_cpt_dataset(cpt_data)
            cpt_model_path = trainer.train_cpt(cpt_dataset, self.config)
            
            ift_dataset = trainer.prepare_ift_dataset(ift_data)
            final_model_path = trainer.train_ift(ift_dataset, cpt_model_path, self.config)
            
            # Evaluate
            evaluator = Evaluator(final_model_path)
            accuracies = evaluator.evaluate_qa_dataset(test_qa)
            
            results.append({
                'num_variations': k,
                'profile_accuracy': accuracies.get('profile', 0),
                'city_accuracy': accuracies.get('city', 0),
                'two_hop_accuracy': accuracies.get('two_hop', 0)
            })
        
        self.results['diversification'] = results
        return results
    
    def run_cot_experiments(self):
        """Run Chain-of-Thought experiments"""
        logger.info("Running CoT experiments...")
        
        profiles = self.data_generator.generate_profiles(1000)
        cities = self.data_generator.generate_city_info([p.city for p in profiles])
        
        train_profiles = profiles[:500]
        test_profiles = profiles[500:]
        
        # Prepare data with and without CoT
        cpt_data = self.data_generator.diversify_cpt_data(train_profiles, cities)
        
        # 50% with CoT, 50% without
        ift_data_cot = self.data_generator.generate_qa_pairs(
            train_profiles[:250], cities, include_cot=True
        )
        ift_data_no_cot = self.data_generator.generate_qa_pairs(
            train_profiles[250:], cities, include_cot=False
        )
        ift_data = ift_data_cot + ift_data_no_cot
        
        # Test data with and without CoT prompting
        test_qa_cot = self.data_generator.generate_qa_pairs(
            test_profiles, cities, include_cot=True
        )
        test_qa_no_cot = self.data_generator.generate_qa_pairs(
            test_profiles, cities, include_cot=False
        )
        
        # Train model
        trainer = ModelTrainer("microsoft/DialoGPT-small", "outputs/cot_experiment")
        
        cpt_dataset = trainer.prepare_cpt_dataset(cpt_data)
        cpt_model_path = trainer.train_cpt(cpt_dataset, self.config)
        
        ift_dataset = trainer.prepare_ift_dataset(ift_data)
        final_model_path = trainer.train_ift(ift_dataset, cpt_model_path, self.config)
        
        # Evaluate both conditions
        evaluator = Evaluator(final_model_path)
        
        acc_cot = evaluator.evaluate_qa_dataset(test_qa_cot)
        acc_no_cot = evaluator.evaluate_qa_dataset(test_qa_no_cot)
        
        results = {
            'with_cot': acc_cot.get('two_hop_cot', 0),
            'without_cot': acc_no_cot.get('two_hop', 0)
        }
        
        self.results['cot'] = results
        return results
    
    def run_forgetting_experiments(self):
        """Run catastrophic forgetting experiments (RQ5)"""
        logger.info("Running forgetting experiments...")
        
        # Create benchmark questions (simulating general knowledge)
        benchmark_qa = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
            {"question": "What is 2 + 2?", "answer": "4"},
            # Add more benchmark questions as needed
        ]
        
        results = []
        
        for dataset_size in [1000, 5000, 10000]:  # Different knowledge scales
            for model_size in ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"]:
                # Evaluate base model first
                base_evaluator = Evaluator(model_size)
                base_score = base_evaluator.evaluate_catastrophic_forgetting(benchmark_qa)
                
                # Train on domain data
                profiles = self.data_generator.generate_profiles(dataset_size)
                cities = self.data_generator.generate_city_info([p.city for p in profiles])
                
                cpt_data = self.data_generator.diversify_cpt_data(profiles, cities)
                ift_data = self.data_generator.generate_qa_pairs(profiles, cities)
                
                trainer = ModelTrainer(model_size, f"outputs/forgetting_{model_size}_{dataset_size}")
                
                cpt_dataset = trainer.prepare_cpt_dataset(cpt_data)
                cpt_model_path = trainer.train_cpt(cpt_dataset, self.config)
                
                ift_dataset = trainer.prepare_ift_dataset(ift_data)
                final_model_path = trainer.train_ift(ift_dataset, cpt_model_path, self.config)
                
                # Evaluate after training
                trained_evaluator = Evaluator(final_model_path)
                trained_score = trained_evaluator.evaluate_catastrophic_forgetting(benchmark_qa)
                
                results.append({
                    'model_size': model_size,
                    'dataset_size': dataset_size,
                    'base_score': base_score,
                    'trained_score': trained_score,
                    'forgetting_ratio': (base_score - trained_score) / base_score if base_score > 0 else 0
                })
        
        self.results['forgetting'] = results
        return results
    
    def generate_plots(self):
        """Generate all plots for the paper"""
        output_dir = Path("plots")
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Scaling laws
        if 'scaling' in self.results:
            self.plot_scaling_laws(output_dir)
        
        # Plot 2: Diversification effects
        if 'diversification' in self.results:
            self.plot_diversification_effects(output_dir)
        
        # Plot 3: CoT effects
        if 'cot' in self.results:
            self.plot_cot_effects(output_dir)
        
        # Plot 4: Forgetting analysis
        if 'forgetting' in self.results:
            self.plot_forgetting_analysis(output_dir)
    
    def plot_scaling_laws(self, output_dir: Path):
        """Generate scaling law plots"""
        df = pd.DataFrame(self.results['scaling'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['profile_accuracy', 'city_accuracy', 'two_hop_accuracy']
        titles = ['Profile Q&A', 'City Q&A', 'Two-hop Queries']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            for dataset_size in df['dataset_size'].unique():
                data_subset = df[df['dataset_size'] == dataset_size]
                axes[i].plot(data_subset['model_size'], data_subset[metric], 
                           marker='o', label=f'{dataset_size}K profiles')
            
            axes[i].set_xlabel('Model Size')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'scaling_laws.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_diversification_effects(self, output_dir: Path):
        """Generate diversification effects plot"""
        df = pd.DataFrame(self.results['diversification'])
        
        plt.figure(figsize=(8, 6))
        
        bars = plt.bar(categories, values, color=['#2E86AB', '#A23B72'], alpha=0.8)
        plt.ylabel('Two-hop Query Accuracy')
        plt.title('Effect of Chain-of-Thought Prompting')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, max(values) * 1.2)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_dir / 'cot_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_forgetting_analysis(self, output_dir: Path):
        """Generate forgetting analysis plots"""
        df = pd.DataFrame(self.results['forgetting'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Forgetting by model size
        model_forgetting = df.groupby('model_size')['forgetting_ratio'].mean()
        ax1.bar(range(len(model_forgetting)), model_forgetting.values)
        ax1.set_xlabel('Model Size')
        ax1.set_ylabel('Average Forgetting Ratio')
        ax1.set_title('Catastrophic Forgetting by Model Size')
        ax1.set_xticks(range(len(model_forgetting)))
        ax1.set_xticklabels([name.split('/')[-1] for name in model_forgetting.index], rotation=45)
        
        # Plot 2: Forgetting by dataset size
        for model in df['model_size'].unique():
            model_data = df[df['model_size'] == model]
            ax2.plot(model_data['dataset_size'], model_data['forgetting_ratio'], 
                    marker='o', label=model.split('/')[-1])
        
        ax2.set_xlabel('Dataset Size')
        ax2.set_ylabel('Forgetting Ratio')
        ax2.set_title('Catastrophic Forgetting by Dataset Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'forgetting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filename}")

