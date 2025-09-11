# Contributing to dplPy

We welcome contributions to dplPy! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, improving documentation, or helping with testing.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.6 or higher
- Git installed and configured
- A GitHub account
- Basic familiarity with dendrochronological concepts (helpful but not required)

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork locally:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dplpy.git
   cd dplpy
   ```

3. **Set up the development environment:**
   ```bash
   # Create conda environment
   conda env create -f environment.yml
   conda activate dplpy
   
   # Install in development mode
   pip install -e .
   
   # Install development tools
   conda install pytest pytest-cov black flake8 mypy jupyter
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Types of Contributions

### 1. Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Environment**: OS, Python version, dplPy version
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected vs actual behavior**
- **Error messages**: Full error traceback if applicable
- **Sample data**: If possible, minimal dataset that triggers the bug

**Template:**
```markdown
## Bug Description
Brief description of the issue.

## Environment
- OS: [e.g., macOS 12.0, Ubuntu 20.04, Windows 10]
- Python version: [e.g., 3.8.5]
- dplPy version: [e.g., 0.1.6]

## Steps to Reproduce
1. Load data with `dpl.readers('file.csv')`
2. Call function `dpl.function(data)`
3. Error occurs

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Error Message
```
Full error traceback here
```

## Sample Data
Attach minimal dataset or provide instructions to reproduce.
```

### 2. Feature Requests

For new features, please provide:

- **Use case**: Why is this feature needed?
- **Proposed functionality**: What should the feature do?
- **API design**: How should users interact with it?
- **Implementation ideas**: Any thoughts on how to implement it
- **References**: Relevant literature or other implementations

### 3. Code Contributions

#### Code Style

We follow these conventions:

- **PEP 8** for Python code style
- **NumPy style** docstrings for all functions
- **Black** for code formatting
- **Type hints** for function parameters and returns (where appropriate)
- **Meaningful variable names** that reflect dendrochronological concepts

Example function structure:
```python
def example_function(data: pd.DataFrame, method: str = "default") -> pd.DataFrame:
    """
    Brief description of what the function does.
    
    Longer description explaining the dendrochronological context,
    methodology, and usage patterns.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Description of the data parameter, including expected structure.
    method : str, optional
        Description of the method parameter, by default "default".
    
    Returns
    -------
    pandas.DataFrame
        Description of what is returned.
    
    Raises
    ------
    ValueError
        Description of when this error is raised.
    
    Notes
    -----
    Additional information about the function, algorithms used,
    limitations, or references to dendrochronological literature.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers('tests/data/csv/ca533.csv')
    >>> result = example_function(data, method="spline")
    >>> print(result.shape)
    (100, 20)
    
    References
    ----------
    .. [1] Author, A. (Year). Title of paper. Journal, vol, pages.
    """
    # Implementation here
    pass
```

#### Testing

All code contributions should include tests:

- **Unit tests** for individual functions
- **Integration tests** for workflows
- **Regression tests** for bug fixes
- **Documentation tests** to ensure examples work

Test file structure:
```
tests/
├── test_readers.py          # Tests for data reading functions
├── test_writers.py          # Tests for data writing functions  
├── test_detrend.py         # Tests for detrending functions
├── test_stats.py           # Tests for statistical functions
└── data/                   # Test data files
    ├── csv/
    └── rwl/
```

Example test:
```python
import pytest
import pandas as pd
import dplpy as dpl

class TestExampleFunction:
    """Test suite for example_function."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_data = dpl.readers('tests/data/csv/ca533.csv')
    
    def test_basic_functionality(self):
        """Test basic function behavior."""
        result = dpl.example_function(self.test_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0
    
    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            dpl.example_function("invalid_input")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = dpl.example_function(empty_df)
        assert result.empty
```

#### Documentation

All functions must have:

- **NumPy-style docstrings** with complete parameter descriptions
- **Usage examples** in the docstring
- **References** to relevant literature when appropriate
- **Type hints** for parameters and returns

### 4. Documentation Contributions

Documentation improvements are always welcome:

- **API documentation**: Improve function docstrings
- **Tutorials**: Create guides for specific analyses
- **Examples**: Add real-world usage examples
- **Theory**: Explain dendrochronological concepts
- **Troubleshooting**: Document common issues and solutions

## Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/descriptive-name

# Or for bug fixes
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/new-feature-name` - For new features
- `fix/bug-description` - For bug fixes
- `docs/documentation-topic` - For documentation
- `test/test-improvements` - For testing improvements

### 2. Make Changes

- Write code following style guidelines
- Add or update tests
- Update documentation
- Run tests locally

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_readers.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Check code style
black src/ tests/
flake8 src/ tests/

# Type checking (if using type hints)
mypy src/
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for negative year values in RWL reader"
git commit -m "Fix cross-dating correlation calculation bug"
git commit -m "Update detrending tutorial with ModNegEx examples"

# Less helpful
git commit -m "Fix bug"
git commit -m "Update docs"
```

### 5. Push and Create Pull Request

```bash
# Push your branch
git push origin feature/descriptive-name

# Create pull request on GitHub
```

## Pull Request Guidelines

### PR Description Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for changes
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #123
Related to #456
```

### Review Process

1. **Automated checks**: All tests and style checks must pass
2. **Code review**: At least one maintainer will review
3. **Discussion**: Address feedback and questions
4. **Approval**: Changes approved by maintainer
5. **Merge**: PR merged into main branch

## Code Review Standards

When reviewing or receiving reviews:

### What We Look For

- **Correctness**: Does the code do what it's supposed to?
- **Completeness**: Are all edge cases handled?
- **Clarity**: Is the code easy to understand?
- **Efficiency**: Is the implementation reasonably efficient?
- **Testing**: Are there adequate tests?
- **Documentation**: Is everything properly documented?
- **Compatibility**: Does it work with existing code?

### Giving Feedback

- Be constructive and specific
- Explain the reasoning behind suggestions
- Acknowledge good code and improvements
- Ask questions to understand intent
- Suggest alternatives when possible

### Receiving Feedback

- Consider all feedback carefully
- Ask for clarification when needed
- Be open to different approaches
- Explain your reasoning when you disagree
- Thank reviewers for their time

## Coding Standards

### Python Code

```python
# Good
def calculate_statistics(data: pd.DataFrame, method: str = "robust") -> dict:
    """Calculate dendrochronological statistics for ring width data."""
    if method not in ["robust", "standard"]:
        raise ValueError(f"Unknown method: {method}")
    
    results = {}
    for column in data.columns:
        series_stats = _calculate_series_stats(data[column], method)
        results[column] = series_stats
    
    return results

# Less ideal
def calc_stats(data, method="robust"):
    results = {}
    for col in data.columns:
        if method == "robust":
            # calculation
            pass
        else:
            # different calculation
            pass
        results[col] = stats
    return results
```

### File Organization

```
src/
├── __init__.py              # Package initialization
├── dplpy.py                # Main interface and CLI
├── readers.py              # Data import functions
├── writers.py              # Data export functions
├── detrend.py              # Detrending methods
├── chron.py                # Chronology building
├── stats.py                # Statistical calculations
├── plot.py                 # Visualization functions
├── xdate.py                # Cross-dating analysis
├── autoreg.py              # Autoregressive modeling
└── utils.py                # Utility functions (if needed)
```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (e.g., 1.0.0): Breaking changes
- **MINOR** (e.g., 0.2.0): New features, backwards compatible
- **PATCH** (e.g., 0.1.1): Bug fixes, backwards compatible

### Changelog

All changes are documented in [CHANGELOG.md](../CHANGELOG.md) following [Keep a Changelog](https://keepachangelog.com/) format.

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge contributions from others
- Be patient with questions and learning

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions and discussion
- **Discussions**: General questions and ideas

## Getting Help

### For Contributors

- Read existing code and tests for examples
- Check the [API documentation](api/index.md)
- Look at recent pull requests for guidance
- Ask questions in GitHub discussions

### For Maintainers

- Review PRs promptly and constructively
- Help new contributors get started
- Keep documentation up to date
- Maintain consistent standards

## Recognition

Contributors are recognized in:

- **CHANGELOG.md**: For significant contributions
- **GitHub contributors list**: Automatic recognition
- **Release notes**: For major features or fixes

Thank you for contributing to dplPy! Your efforts help advance dendrochronological research and make tree-ring analysis more accessible to the scientific community.