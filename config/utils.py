def create_experiment_config() -> ExperimentConfig:
    """Create default experiment configuration"""
    return ExperimentConfig(
        dataset_sizes=[1000, 5000, 10000],  # Reduced for demo purposes
        model_sizes=["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"],  # Using smaller models for demo
        num_variations=[1, 5, 10],
        vocab_sizes=[32000, 128000],  # Simulated vocab sizes
        batch_size=8,  # Reduced for memory constraints
        max_length=1024,  # Reduced for efficiency
        learning_rate_cpt=1e-4,
        learning_rate_ift=5e-5,
        num_epochs_cpt=1,  # Reduced for demo
        num_epochs_ift=1,  # Reduced for demo
        output_dir="outputs"
    )

