# Master Thesis: AI Agents and Distributive Justice

## Overview

This repository contains the implementation for Lucas Mueller's master thesis, which replicates the distributive justice experiments from Frohlich & Oppenheimer (1992) "Choosing Justice: An Experimental Approach to Ethical Theory" using AI agents. The framework simulates a "veil of ignorance" scenario where AI agents engage in structured discussion to reach consensus on principles of distributive justice through formal voting mechanisms.

The folder `hypothesis_testing` contains the execution of the testing of the hypothesis presented in the thesis. The remaining folders contain files related to the experiment itself. On a high level, the application (`main.py`) takes in a `.yaml` file outlining the configuration (e.g., `default.yaml`), executes the experiments and returns a structured log file as `.JSON` and optionally the entire raw transcript of all agents interactions.

## Architecture

The main application relies on the OpenAI Agents SDK (open source), and employs a service based architecture. 
See [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md):  for detailed architecture documentation, service ownership guides, and testing strategies.

## Key Features

- **Two-phase experiment structure**: Exact replication of Frohlich & Oppenheimer (1992) with two phases: Individual familiarization (Phase 1) and group discussion (Phase 2)
- **Multi-language support**: English, Spanish, and Mandarin experiments with localized prompts
- **Multi-provider support**: OpenAI, Google Gemini, OpenRouter, and local Ollama models
- **Reproducible experiments**: Comprehensive seeding for deterministic results
- **Services-first architecture**: Clean separation of concerns with protocol-based dependency injection

## Installation

### Requirements

- Python 3.11 or higher
- OpenAI API key, or at least one of the following: Gemini API key for Google models, OpenRouter API key for multi-provider access, or locally hosted LLM through Ollama

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:
```bash
# Option 1:
OPENAI_API_KEY=your_openai_key_here

# Option 2
GEMINI_API_KEY=your_gemini_key_here

# Option 3 
OPENROUTER_API_KEY=your_openrouter_key_here

# Option 4: Running local Ollama OpenAI API chat completions style endpoint

# Optional: Control OpenAI Agents SDK tracing
OPENAI_AGENTS_DISABLE_TRACING=1
```

## Running Experiments

### Basic Usage

```bash
# Run with a configuration file (required)
python main.py config/fast.yaml

# Specify custom output path
python main.py config/fast.yaml results/my_experiment.json
```

### Configuration Files

Example configurations are available in the `config/` directory.

### Configuration Structure

Experiment configurations are YAML files defining:
- Agent properties (name, model, language, temperature)
- Phase 2 settings (discussion rounds, voting parameters, memory management)
- Income distribution settings and probabilities
- Reproducibility seed for deterministic experiments

See çfor detailed configuration documentation.

## Project Structure

```
Master_Thesis_Lucas_Mueller/
├── config/                   # Experiment configuration files (YAML)
├── core/                     # Core experiment orchestration
│   ├── services/             # Phase 2 services architecture
│   ├── experiment_manager.py
│   ├── phase1_manager.py
│   └── phase2_manager.py
├── experiment_agents/        # Agent implementations
├── models/                   # Data models and types
├── utils/                    # Utility modules
├── translations/             # Multilingual prompt templates
├── tests/                    # Test suite
├── hypothesis_testing/       # Organized experimental conditions
├── docs/                     # Documentation and diagrams
├── main.py                   # Experiment runner entry point
├── TECHNICAL_ARCHITECTURE.md # Detailed technical documentation
└── requirements.txt          # Python dependencies
```

## Documentation

- [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md): Comprehensive technical documentation with architectural diagrams
- [docs/diagrams/](docs/diagrams/): Visual architectural guides (9 progressive diagrams)


## Citation

This implementation is based on:

Frohlich, N., & Oppenheimer, J. A. (1992). *Choosing Justice: An Experimental Approach to Ethical Theory*. University of California Press.

## License

This repository is part of a master's thesis. Please contact Lucas Mueller for usage permissions.


