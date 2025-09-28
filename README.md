# Sherlock Holmes Text Generator

A neural language model that generates text in the style of Sherlock Holmes stories using LSTM with attention mechanism.

## Quick Start

1. **Install dependencies:**
```bash
uv sync
```

2. **Train the model:**
```bash
uv run python train.py
```

3. **Generate text:**
```bash
uv run python predict.py "Holmes said" 15
uv run python predict.py "I saw Watson" 10
```

## Usage

**Command line:**
```bash
uv run python predict.py "your prompt" [num_tokens]
```

**Interactive mode:**
```bash
uv run python predict.py
```

The model will utilise the saved model as part of repo to generate Sherlock Holmes-style text.
