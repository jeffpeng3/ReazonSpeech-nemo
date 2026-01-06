# Developer Guidelines for ReazonSpeech-nemo

This document provides instructions for agents and developers working on the ReazonSpeech-nemo repository.

## 1. Environment & Build

### Installation
This project uses `pyproject.toml` for configuration.

```bash
# Install in editable mode with dependencies
pip install -e .
```

### Running the Application
The package exposes a CLI tool defined in `pyproject.toml`:

```bash
# Run the CLI
reazonspeech-nemo-asr --help
```

### Dependencies
Key dependencies include:
- `torch`
- `nemo_toolkit[asr]`
- `numpy`, `librosa`, `soundfile`

## 2. Testing

**Current Status:** No test suite is currently present in the repository.

### Adding Tests
- Use **pytest** for any new tests.
- Place tests in a `tests/` directory at the root or alongside source files.
- Name test files `test_*.py` or `*_test.py`.

### Running Tests (Future)
Once tests are added, use the following commands:

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_transcribe.py

# Run a specific test function
pytest tests/test_transcribe.py::test_load_model
```

## 3. Linting & Formatting

No specific linter configuration is present (`ruff.toml` or `.flake8` are missing).
However, follow standard Python conventions (PEP 8).

Recommended commands for verification:
```bash
# Check for linting errors (assuming ruff is installed)
ruff check src/

# Format code
ruff format src/
```

## 4. Code Style Guidelines

### General
- **Language:** Python 3.
- **Indentation:** 4 spaces.
- **Line Length:** Standard 88 or 120 characters.

### Imports
Organize imports in the following order:
1. Standard Library (e.g., `dataclasses`)
2. Third-Party Libraries (e.g., `torch`, `numpy`, `nemo`)
3. Local/Relative Imports (e.g., `from .interface import ...`)

**Example:**
```python
import torch
from dataclasses import dataclass

import numpy as np
from nemo.utils import logging

from .interface import TranscribeConfig
```

### Typing
- **Type Hints:** Mandatory for function arguments and return values.
- **Dataclasses:** Use `@dataclass` for data structures instead of raw dictionaries or tuples.

**Example:**
```python
def transcribe(model: EncDecRNNTBPEModel, audio: AudioData) -> TranscribeResult:
    ...
```

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `TranscribeResult`, `AudioData`).
- **Functions/Methods:** `snake_case` (e.g., `load_model`, `decode_hypothesis`).
- **Variables:** `snake_case`.
- **Constants:** `UPPER_CASE` (e.g., `PAD_SECONDS`).

### Docstrings
Use Google-style docstrings for functions and classes.

**Example:**
```python
def transcribe(model, audio, config=None):
    """Inference audio data using NeMo model

    Args:
        model (nemo.collections.asr.models.EncDecRNNTBPEModel): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult: The result of the transcription.
    """
```

### Error Handling
- Use explicit exception classes (e.g., `RuntimeError` for missing GPU).
- Validate inputs early (e.g., checking if `config` is `None`).

**Example:**
```python
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available, please use CPU version of ReazonSpeech")
```

## 5. Agent Instructions

When acting as an AI agent on this codebase:

1.  **Conventions:** Strictly adhere to the coding style observed in `src/`. Mimic the use of `dataclasses` and explicit typing.
2.  **No Tests:** Be aware that no tests currently exist. If you write code, you **must** verify it by creating a small reproduction script or a new unit test if possible.
3.  **GPU/CPU:** Note that `src/transcribe.py` checks for CUDA. When mocking or writing tests, ensure logic handles CPU environments or mocks `torch.cuda.is_available()`.
4.  **Dependencies:** Do not introduce new heavy dependencies without verification. The project relies heavily on `nemo_toolkit`.
5.  **Pathing:** Always use absolute paths for file operations.
