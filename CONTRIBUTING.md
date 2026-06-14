# Contributing to Momentum-Research

Thank you for your interest in contributing to this research project. This document outlines the guidelines for reporting issues, submitting changes, and maintaining code quality.

## How to Report Issues

- **Bug reports:** Open a [GitHub issue](https://github.com/dshan12/Momentum-Research/issues) with a clear title, description of expected vs. actual behavior, and steps to reproduce. Include your Python version and OS.
- **Data errors:** If you identify inaccuracies in the S&P 500 membership history or price data, please provide the specific ticker, date, and source for the correction.
- **Feature requests:** Open an issue with the "enhancement" tag describing the proposed change and its motivation.

## How to Submit Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, following the coding standards below.
3. Add or update tests as needed.
4. Run the test suite locally.
5. Submit a pull request with a clear description of the changes and any relevant issue numbers.

**PR checklist before submission:**

- [ ] Code passes `mypy src/`
- [ ] All tests pass (`pytest`)
- [ ] New functionality includes tests
- [ ] Type hints are present on all function signatures

## Coding Standards

### General

- Line length: 88 characters
- Quote style: double quotes
- Follow existing patterns in the codebase for consistency

### Type Hints

All function signatures must include type hints. This is enforced by mypy in strict mode.

```python
def compute_sharpe(returns: pd.Series, rf_annual: float = 0.02) -> float:
    ...
```

### Docstrings

Use descriptive comments for non-obvious logic. Full docstrings are not required for research code, but complex functions should include a brief explanation of inputs, outputs, and any assumptions.

### Testing

- Tests are located in `tests/` and use pytest.
- Use the fixtures in `conftest.py` where applicable.
- Aim for coverage of core logic (turnover model, return construction, weight calculations).
- Tests that require network access are avoided; use synthetic data in fixtures.

### Imports

Group imports in the following order with a blank line between each group:

1. Standard library (`os`, `typing`, etc.)
2. Third-party libraries (`numpy`, `pandas`, `statsmodels`, etc.)
3. Local application (`data.turnover`, etc.)

## Running Tests Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_turnover.py

# Run with verbose output
pytest -v --tb=long
```

## Data Pipeline Notes

- The `data/cleaned/` directory contains pre-computed outputs. Commit updated data files only when the analysis has been validated.
- The `archive/` directory contains legacy scripts for reference but is not part of the active pipeline.
- New data sources should be added as modules in `src/data/` and integrated into the pipeline scripts.
