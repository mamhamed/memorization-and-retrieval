# Memorization and Retrieval Experimentation
This repository contains the complete experimental framework for the paper "Recall at Scale: Investigating Knowledge Storage, Retrieval, and Forgetting in LLMs".

research on how an LLM store information and under what condition it can retrieve that in    formation

## Overview
This framework implements comprehensive experiments to study how Large Language Models store, retrieve, and forget knowledge at scale. The experiments cover:

Scaling Laws: How model size and data size affect recall performance
Data Diversification: Impact of format variations on knowledge retention
Chain-of-Thought: Effectiveness of CoT prompting for two-hop queries
Catastrophic Forgetting: How new knowledge affects previously learned information
Vocabulary Size: Impact of tokenizer vocabulary size on recall
Profile Length: How content length affects memorization

## Quick Start
### 1. Setup Environment

```bash
# Make the main script executable
chmod +x run_experiments.sh
# Run with automatic environment setup
./run test
```
This will:
* Create a Python environment with all dependencies
* Generate test data
* Run a minimal experiment to validate the setup

### 2. Validate Setup
```bash
# Run comprehensive validation
python test_setup.py --test all

# Or test specific components
python test_setup.py --test data
python test_setup.py --test model
```

3. Run Experiments
```bash
# Run all experiments
./run.sh all

# Run specific experiments
./run.sh scaling
./run.sh cot
./run.sh forgetting
./run.sh vocabulary vocab_config.yaml 
```
