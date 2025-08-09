#!/bin/bash

# run.sh - Complete experimental pipeline for LLM Knowledge Recall research
# Usage: ./run.sh [experiment_type] [config_file]


# Default values
EXPERIMENT_TYPE=${1:-"all"}
CONFIG_FILE=${2:-"config.yaml"}
PYTHON_ENV=${3:-"llm_recall_env"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Check if conda is available 
    if command_exists python3 && command_exists pip3; then
        info "Using virtual environment with pip3"
        
        # Create virtual environment if it doesn't exist
        if [[ ! -d "${PYTHON_ENV}" ]]; then
            log "Creating virtual environment: ${PYTHON_ENV}"
            python3 -m venv "${PYTHON_ENV}"
	    # log "Installing required packages..."
	    pip3 install --upgrade pip
	    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	    pip3 install transformers==4.36.0
	    pip3 install datasets==2.15.0
	    pip3 install accelerate==0.25.0
	    pip3 install deepspeed==0.12.6
	    pip3 install wandb==0.16.1
	    pip3 install matplotlib==3.8.2
	    pip3 install seaborn==0.13.0
	    pip3 install pandas==2.1.4
	    pip3 install numpy==1.24.4
	    pip3 install scikit-learn==1.3.2
	    pip3 install scipy==1.11.4
	    pip3 install plotly==5.17.0
	    pip3 install faker==20.1.0
	    pip3 install jinja2==3.1.2
	    pip3 install pyyaml==6.0.1
	    pip3 install tqdm==4.66.1

        fi
        
        # Activate environment
        source "${PYTHON_ENV}/bin/activate"
        
    else
        error "Neither conda nor python3/pip found. Please install one of them."
        exit 1
    fi
    
    # Install required packages
    # log "Installing required packages..."
    pip3 install --upgrade pip
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip3 install transformers==4.36.0
    pip3 install datasets==2.15.0
    pip3 install accelerate==0.25.0
    pip3 install deepspeed==0.12.6
    pip3 install wandb==0.16.1
    pip3 install matplotlib==3.8.2
    pip3 install seaborn==0.13.0
    pip3 install pandas==2.1.4
    pip3 install numpy==1.24.4
    pip3 install scikit-learn==1.3.2
    pip3 install scipy==1.11.4
    pip3 install plotly==5.17.0
    pip3 install faker==20.1.0
    pip3 install jinja2==3.1.2
    pip3 install pyyaml==6.0.1
    pip3 install tqdm==4.66.1
    
    log "Environment setup completed successfully!"
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check for CUDA
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        info "Found ${GPU_COUNT} GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        warning "CUDA not detected. Experiments will run on CPU (much slower)."
    fi
    
    # Check available memory
    if command_exists free; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        info "Total system memory: ${TOTAL_MEM}GB"
        
        if [[ ${TOTAL_MEM} -lt 16 ]]; then
            warning "Less than 16GB RAM detected. Large model experiments may fail."
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2{print $4}')
    info "Available disk space: ${AVAILABLE_SPACE}"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    mkdir -p outputs/{models,data,logs}
    mkdir -p plots/{scaling,diversification,cot,forgetting,vocabulary,length}
    mkdir -p advanced_plots
    mkdir -p data/{synthetic,processed}
    mkdir -p results
    
    log "Directory structure created."
}

# Generate default configuration if not exists
generate_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log "Generating default configuration file: ${CONFIG_FILE}"
        
        cat > "${CONFIG_FILE}" << EOF
# Configuration for LLM Knowledge Recall Experiments

# Dataset configurations  
dataset_sizes: [1000, 5000]  # Start small for testing
model_sizes:
  - "microsoft/DialoGPT-small"    # ~117M parameters (for testing)
  - "microsoft/DialoGPT-medium"   # ~345M parameters

# Data diversification
num_variations: [1, 5, 10]

# Training hyperparameters
batch_size: 4  # Small batch size for testing
max_length: 1024  # Reduced for testing
learning_rate_cpt: 1e-4
learning_rate_ift: 5e-5
num_epochs_cpt: 2  # Reduced for testing
num_epochs_ift: 1  # Reduced for testing

# Output configuration
output_dir: "outputs"

# Hardware configuration
gradient_accumulation_steps: 8
max_grad_norm: 1.0
warmup_steps: 100

# Evaluation configuration
eval_batch_size: 8
max_eval_samples: 500

# Experiment-specific settings
scaling_experiment:
  enable: true
  
diversification_experiment:
  enable: true
  
cot_experiment:
  enable: true
  
forgetting_experiment:
  enable: true
  
vocabulary_experiment:
  enable: false  # Disabled by default (requires large models)
  
profile_length_experiment:
  enable: true
EOF
        
        info "Default configuration created. Edit ${CONFIG_FILE} to customize experiments."
    fi
}

# Run data generation
run_data_generation() {
    log "Generating synthetic data..."
    
    python3 -c "
from data.data_generator import DataGenerator
import json

# Generate sample data
generator = DataGenerator()
profiles = generator.generate_profiles(1000)
cities = generator.generate_city_info([p.city for p in profiles])

# Save sample data for inspection
sample_data = {
    'sample_profiles': [profiles[i].__dict__ for i in range(5)],
    'sample_cities': [cities[i].__dict__ for i in range(3)],
    'stats': {
        'total_profiles': len(profiles),
        'unique_cities': len(cities),
        'sample_cpt_data': len(generator.diversify_cpt_data(profiles[:10], cities[:5], 5))
    }
}

with open('data/sample_data_info.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print('Sample data generated successfully!')
"
    
    log "Data generation completed."
}

# Run main experiments
run_main_experiments() {
    log "Starting main experiments..."
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    
    # Run experiments based on type
    case ${EXPERIMENT_TYPE} in
        "scaling")
            info "Running scaling law experiments..."
            python3 -m llm_knowledge_recall --experiment scaling --config "${CONFIG_FILE}"
            ;;
        "diversification")
            info "Running data diversification experiments..."
            python3 -m llm_knowledge_recall --experiment diversification --config "${CONFIG_FILE}"
            ;;
        "cot")
            info "Running Chain-of-Thought experiments..."
            python3 -m llm_knowledge_recall --experiment cot --config "${CONFIG_FILE}"
            ;;
        "forgetting")
            info "Running catastrophic forgetting experiments..."
            python3 -m llm_knowledge_recall --experiment forgetting --config "${CONFIG_FILE}"
            ;;
        "vocabulary")
            info "Running vocabulary size experiments..."
            python3 -m main --experiment vocab --config "${CONFIG_FILE}"
            ;;
        "length")
            info "Running profile length experiments..."
            python3 -m main --experiment length --config "${CONFIG_FILE}"
            ;;
        "all")
            info "Running all experiments..."
            python3 -m llm_knowledge_recall --experiment all --config "${CONFIG_FILE}"
            ;;
        *)
            error "Unknown experiment type: ${EXPERIMENT_TYPE}"
            echo "Available types: scaling, diversification, cot, forgetting, vocabulary, length, all"
            exit 1
            ;;
    esac
}

# Run advanced analysis
run_advanced_analysis() {
    log "Running advanced analysis..."
    
    if [[ -f "final_experiment_results.json" ]]; then
        python3 -c "
import sys
sys.path.append('.')
from data_analysis_utils import run_advanced_analysis
run_advanced_analysis('final_experiment_results.json')
"
        log "Advanced analysis completed."
    else
        warning "No results file found. Skipping advanced analysis."
    fi
}

# Generate final report
generate_report() {
    log "Generating final report..."
    
    REPORT_FILE="experiment_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "${REPORT_FILE}" << EOF
# LLM Knowledge Recall Experiments Report

**Generated:** $(date)
**Experiment Type:** ${EXPERIMENT_TYPE}
**Configuration:** ${CONFIG_FILE}

## Experiment Summary

This report summarizes the results of experiments investigating knowledge storage, retrieval, and forgetting in Large Language Models.

## System Information

- **GPU Count:** ${GPU_COUNT:-0}
- **Total Memory:** ${TOTAL_MEM:-Unknown}GB
- **Available Disk Space:** ${AVAILABLE_SPACE:-Unknown}

## Files Generated

### Results
- \`final_experiment_results.json\` - Raw experimental results
- \`statistical_report.txt\` - Statistical analysis report

### Visualizations
- \`plots/\` - Basic experiment plots
- \`advanced_plots/\` - Comprehensive analysis plots
- \`interactive_scaling_laws.html\` - Interactive scaling visualizations

### Data
- \`data/sample_data_info.json\` - Information about generated data
- \`outputs/\` - Trained models and intermediate outputs

## Key Findings

$(if [[ -f "statistical_report.txt" ]]; then
    echo "### Statistical Summary"
    echo "\`\`\`"
    head -n 50 statistical_report.txt
    echo "\`\`\`"
    echo ""
    echo "*See statistical_report.txt for complete analysis*"
else
    echo "*Statistical analysis not available - check if experiments completed successfully*"
fi)

## Next Steps

1. Review the generated plots in \`plots/\` and \`advanced_plots/\`
2. Examine detailed results in \`final_experiment_results.json\`
3. Check \`statistical_report.txt\` for statistical significance tests
4. Use the interactive plots (\`interactive_scaling_laws.html\`) for deeper exploration

## Reproduction

To reproduce these results:

\`\`\`bash
./run_experiments.sh ${EXPERIMENT_TYPE} ${CONFIG_FILE}
\`\`\`

---
*Report generated by LLM Knowledge Recall Experiment Pipeline*
EOF

    log "Report generated: ${REPORT_FILE}"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary model files if space is needed
    if [[ -d "outputs/models" ]]; then
        find outputs/models -name "*.bin" -size +1G -delete 2>/dev/null || true
        find outputs/models -name "pytorch_model.bin" -exec rm -f {} \; 2>/dev/null || true
    fi
    
    # Clean up cache
    if command_exists pip; then
        pip cache purge >/dev/null 2>&1 || true
    fi
    
    log "Cleanup completed."
}

# Error handling
handle_error() {
    local line_number=$1
    error "An error occurred on line ${line_number}"
    error "Experiment pipeline failed!"
    
    # Try to save partial results
    if [[ -f "outputs/partial_results.json" ]]; then
        cp outputs/partial_results.json "failed_experiment_results_$(date +%Y%m%d_%H%M%S).json"
        info "Partial results saved."
    fi
    
    cleanup
    exit 1
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# Main execution function
main() {
    log "Starting LLM Knowledge Recall Experiment Pipeline"
    log "Experiment type: ${EXPERIMENT_TYPE}"
    log "Configuration file: ${CONFIG_FILE}"
    
    # Check if running in test mode
    if [[ "${EXPERIMENT_TYPE}" == "test" ]]; then
        log "Running in TEST mode - using minimal configurations"
        export TEST_MODE=1
    fi
    
    # Pipeline steps
    # check_system_requirements # uncomment if first time
    setup_environment
    create_directories
    generate_config
    run_data_generation
    run_main_experiments
    run_advanced_analysis
    generate_report
    cleanup
    
    log "Experiment pipeline completed successfully!"
    log "Check the generated report and results files."
    
    # Final summary
    echo ""
    echo "=== EXPERIMENT COMPLETED ==="
    echo "Results available in:"
    echo "  - Final report: experiment_report_*.md"
    echo "  - Raw results: final_experiment_results.json"
    echo "  - Plots: plots/ and advanced_plots/ directories"
    echo "  - Statistical analysis: statistical_report.txt"
    echo ""
    echo "To view interactive plots, open: interactive_scaling_laws.html"
    echo "============================"
}

# Help function
show_help() {
    echo "LLM Knowledge Recall Experiment Pipeline"
    echo ""
    echo "Usage: $0 [EXPERIMENT_TYPE] [CONFIG_FILE] [PYTHON_ENV]"
    echo ""
    echo "EXPERIMENT_TYPE (default: all):"
    echo "  scaling        - Run scaling law experiments"
    echo "  diversification - Run data diversification experiments"
    echo "  cot            - Run Chain-of-Thought experiments"
    echo "  forgetting     - Run catastrophic forgetting experiments"
    echo "  vocabulary     - Run vocabulary size experiments"
    echo "  length         - Run profile length experiments"
    echo "  all            - Run all experiments"
    echo "  test           - Run quick test with minimal configuration"
    echo ""
    echo "CONFIG_FILE (default: config.yaml):"
    echo "  Path to YAML configuration file"
    echo ""
    echo "PYTHON_ENV (default: llm_recall_env):"
    echo "  Name of Python environment to create/use"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run all experiments with default config"
    echo "  $0 scaling                   # Run only scaling experiments"
    echo "  $0 test                      # Quick test run"
    echo "  $0 all custom_config.yaml   # Use custom configuration"
    echo ""
    echo "Environment Variables:"
    echo "  CUDA_VISIBLE_DEVICES  - Specify which GPUs to use"
    echo "  WANDB_PROJECT         - W&B project name for logging"
    echo "  HF_TOKEN             - Hugging Face token for private models"
    echo ""
}

# Check if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"

