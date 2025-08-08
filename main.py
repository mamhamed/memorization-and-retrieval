import argparse
import yaml
from pathlib import Path

from config.experiment import ExperimentConfig
from model.experiment_runner import ExperimentRunner
from model.vocab_experiment import VocabExperiment


import os
os.environ["WANDB_DISABLED"] = "true"

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run LLM Knowledge Recall Experiments")
    parser.add_argument("--experiment", choices=['all', 'scaling', 'diversification', 'cot', 'forgetting', 'vocab', 'length'], 
                       default='all', help="Which experiment to run")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(config_dict)
    else:
        # Default configuration
        config_dict = {
            'dataset_sizes': [1000, 5000],
            'model_sizes': ['microsoft/DialoGPT-small', 'microsoft/DialoGPT-medium'],
            'num_variations': [1, 5, 10],
            'vocab_sizes': [32000, 128000],
            'batch_size': 8,
            'max_length': 2048,
            'learning_rate_cpt': 1e-4,
            'learning_rate_ift': 5e-5,
            'num_epochs_cpt': 1,
            'num_epochs_ift': 1,
            'output_dir': 'outputs'
        }
        
        # Save default config
        with open(args.config, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    config_dict_tmp = config_dict.copy() # mhf
    config_dict_tmp.pop('vocabulary_experiment') #mhf
    config = ExperimentConfig(**config_dict_tmp) #mhf
    
    # Create output directories
    Path(config.output_dir).mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Run specified experiments
    if args.experiment == 'all' or args.experiment == 'scaling':
        runner.run_scaling_experiments()
    
    if args.experiment == 'all' or args.experiment == 'diversification':
        runner.run_diversification_experiments()
    
    if args.experiment == 'all' or args.experiment == 'cot':
        runner.run_cot_experiments()
    
    if args.experiment == 'all' or args.experiment == 'forgetting':
        runner.run_forgetting_experiments()
    
    if args.experiment == 'vocab':
        vocab_exp = VocabExperiment(config)
        vocab_results = vocab_exp.run_vocabulary_comparison()
        runner.results['vocabulary'] = vocab_results
    
    if args.experiment == 'length':
        # Use a sample tokenizer for length analysis
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        length_analyzer = ProfileLengthAnalyzer(tokenizer)
        length_results = length_analyzer.run_length_analysis()
        runner.results['profile_length'] = length_results
    
    # Generate plots and save results
    runner.generate_plots()
    runner.save_results("final_experiment_results.json")
    
    logger.info("All experiments completed successfully!")
    logger.info("Results saved to 'final_experiment_results.json'")
    logger.info("Plots saved to 'plots/' directory")

if __name__ == "__main__":
    main()
