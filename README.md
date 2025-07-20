
# pandas-and-numpy

Analyze survey results with Python, pandas, numpy, and machine learning tools.

## Features
- Data cleaning and summary statistics
- Correlation heatmap visualization
- Salary prediction (if available)
- Top countries and languages analysis

## Setup
Install dependencies and create a virtual environment:

```sh
poetry install
```

## Usage
Run the analyzer script:

```sh
poetry run python survey_results_analyzer.py
```

## Testing
Run all tests:

```sh
poetry run pytest
```

## Development Tools
- Type checking: `poetry run mypy survey_results_analyzer.py`
- Linting: `poetry run ruff check .`
- Pre-commit hooks: `poetry run pre-commit install`

## Project Structure
- `survey_results_analyzer.py` — Main analysis script
- `survey_results_public.csv` — Survey data file
- `tests/` — Unit tests

## Contributing
Feel free to open issues or submit pull requests for improvements!
