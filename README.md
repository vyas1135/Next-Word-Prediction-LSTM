# Sherlock Holmes Next Word Prediction

A neural language model that predicts next word in the style of Sherlock Holmes stories using LSTM with attention mechanism.

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
uv run python predict.py "<Seed-Phrase>" 15
uv run python predict.py "<Seed-Phrase>" 10
```

## Usage

**Command line:**
```bash
uv run python predict.py "your prompt" [num_generated_words]
```

**Interactive mode:**
```bash
uv run python predict.py
```

The model will utilise the saved model as part of repo to generate Sherlock Holmes-style text.
