# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate.sh

# Install project and dependencies
uv pip install .
# Or for development with all extras:
make install
```

### Testing
```bash
# Run full test suite
make test
# Or directly:
uv run --all-extras pytest

# Run specific test file
uv run --all-extras pytest tests/test_predictions.py

# Run with coverage
make coverage
```

### Code Quality
```bash
# Run linting and formatting
make lint
# Or directly:
uv run --all-extras ruff check marginaleffects
uv run --all-extras ruff format marginaleffects

# Run pre-commit hooks
make precommit
```

### Documentation
```bash
# Extract docstrings into Quarto files
make qmd
# Inject documentation
make inject_docs
```

### Build and Publish
```bash
# Build package
make build
# Publish (after build)
make publish
```

## Architecture Overview

This is a Python package for statistical marginal effects analysis, providing unified interfaces for predictions, comparisons (contrasts), and slopes across multiple statistical modeling frameworks.

### Core Components

**Main API Functions** (in `marginaleffects/__init__.py`):
- `predictions()` / `avg_predictions()` - Generate predictions/fitted values
- `comparisons()` / `avg_comparisons()` - Compute contrasts and differences  
- `slopes()` / `avg_slopes()` - Calculate marginal effects/partial derivatives
- `hypotheses()` - Hypothesis testing framework
- `datagrid()` - Create reference grids for analysis
- Plot functions: `plot_predictions()`, `plot_comparisons()`, `plot_slopes()`

**Model Adapters** (files starting with `model_`):
- `model_statsmodels.py` - StatsModels integration via `fit_statsmodels()`
- `model_sklearn.py` - Scikit-learn integration via `fit_sklearn()` 
- `model_linearmodels.py` - LinearModels integration via `fit_linearmodels()`
- `model_pyfixest.py` - PyFixest integration
- `model_abstract.py` - Abstract base class defining model interface

**Core Infrastructure**:
- `classes.py` - `MarginaleffectsDataFrame` extends Polars DataFrame with metadata
- `sanitize_model.py` - Model wrapper/adapter logic
- `uncertainty.py` - Standard error calculations, jacobians, confidence intervals
- `transform.py` - Transformations (log, logit, etc.)
- `by.py` - Grouping/stratification logic
- `validation.py` - Input validation and type checking

### Key Design Patterns

1. **Adapter Pattern**: Model-specific adapters (`model_*.py`) provide unified interface across different modeling libraries (statsmodels, sklearn, etc.)

2. **Polars-Based Data Handling**: Uses Polars DataFrames throughout with custom `MarginaleffectsDataFrame` wrapper for metadata

3. **Functional Composition**: Core functions like `slopes()` compose `comparisons()` with different parameters rather than reimplementing logic

4. **Extensible Model Support**: Adding new model types requires implementing the abstract interface in `model_abstract.py`

### Testing Strategy

- Comprehensive test suite in `tests/` with model-specific test files
- Cross-validation against R implementation using reference data in `tests/r/`
- Visual regression testing for plots with reference images in `tests/images/`
- Uses pytest with optional extras for different modeling libraries

### Dependencies

- **Core**: polars, numpy, scipy, formulaic, patsy, plotnine, pydantic, pyarrow
- **Testing**: statsmodels, sklearn, linearmodels, pyfixest (as optional test dependencies)
- **Dev**: pytest, ruff, pre-commit, uv for dependency management