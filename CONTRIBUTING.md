# Contributing to MCD-rPPG

We welcome contributions to the MCD-rPPG project! This document outlines how you can contribute to the repository.

## 📋 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [How to Contribute](#-how-to-contribute)
3. [Development Setup](#-development-setup)
4. [Pull Request Guidelines](#-pull-request-guidelines)
5. [Reporting Issues](#-reporting-issues)
6. [Coding Standards](#-coding-standards)
7. [Testing](#-testing)
8. [Documentation](#-documentation)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).

Be respectful, inclusive, and collaborative. Harassment or discriminatory behavior will not be tolerated.

## 🚀 How to Contribute

### Ways to Contribute

1. **Report Bugs** - Submit issues for bugs you find
2. **Suggest Features** - Propose new features or improvements
3. **Fix Bugs** - Submit pull requests to fix issues
4. **Add Features** - Implement new functionality
5. **Improve Documentation** - Help improve docs and examples
6. **Review PRs** - Review pull requests from other contributors
7. **Share Models** - Share trained models or configurations

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** and test them
5. **Submit a pull request** to the main repository

## 💻 Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/mcd_rppg.git
cd mcd_rppg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If exists

# Install in development mode
pip install -e .
```

### Development Dependencies

```bash
# Testing
pip install pytest pytest-cov

# Linting
pip install flake8 black isort mypy

# Documentation
pip install mkdocs mkdocs-material
```

## 📝 Pull Request Guidelines

### Before Submitting

1. **Check for existing issues** - Search GitHub issues to avoid duplicates
2. **Create an issue first** - For significant changes, discuss in an issue first
3. **Fork the repository** - Don't work on the main repository directly
4. **Create a feature branch** - Use descriptive branch names

### Branch Naming

Use the following branch naming conventions:

- `feature/short-description` - For new features
- `fix/short-description` - For bug fixes
- `docs/short-description` - For documentation changes
- `refactor/short-description` - For code refactoring
- `perf/short-description` - For performance improvements

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): subject

body

footer
```

Examples:
- `feat(preprocessing): add face alignment support`
- `fix(dataset): correct synchronization offset calculation`
- `docs(readme): update installation instructions`
- `perf(training): optimize data loading`

### Pull Request Requirements

1. **Clear title** - Descriptive and concise
2. **Detailed description** - Explain what and why
3. **Linked issues** - Reference related issues
4. **Tests pass** - All existing tests must pass
5. **Code quality** - Follow coding standards
6. **Documentation** - Update docs if needed
7. **Squashed commits** - Clean commit history

### PR Template

```markdown
## Description

[Clear description of the changes]

## Related Issues

[Link to related issues]

## Changes Made

- [List of changes]

## Testing

[How you tested the changes]

## Checklist

- [ ] Code follows project standards
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Reporting Issues

### Before Reporting

1. **Search existing issues** - Check if the issue already exists
2. **Check documentation** - Review README, DATASET.md, etc.
3. **Try latest version** - Ensure you're using the latest code

### Issue Template

```markdown
## Description

[Clear description of the issue]

## Steps to Reproduce

1. [First step]
2. [Second step]
3. [Third step]

## Expected Behavior

[What should happen]

## Actual Behavior

[What actually happens]

## Environment

- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.12]
- Package versions: [e.g., torch==2.6.0, face_alignment==1.4.1]
- Hardware: [e.g., RTX 4090, 64GB RAM]

## Additional Information

[Any other relevant information]
```

### Issue Labels

Issues may have the following labels:

- `bug` - Confirmed bug
- `enhancement` - Feature request
- `documentation` - Documentation improvement
- `good first issue` - Suitable for new contributors
- `help wanted` - Needs community help
- `preprocessing` - Related to preprocessing
- `training` - Related to training
- `inference` - Related to inference
- `dataset` - Related to dataset

## 📏 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to functions and classes

### Type Hints

Use type hints for better code clarity:

```python
def process_video(video: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Process video with face landmarks."""
    pass
```

### Imports

- Group imports by type (standard, third-party, local)
- Sort imports alphabetically within groups
- Use absolute imports for local modules

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# Local
from rppglib import face_utils
from rppglib import processing
```

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Document all public functions and classes
- Include type information in docstrings

```python
def align_signals(video: np.ndarray, ppg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align video frames with PPG signal.
    
    Args:
        video: Array of video frames with shape (T, H, W, 3).
        ppg: Array of PPG signal with shape (T,).
    
    Returns:
        Tuple of (aligned_video, aligned_ppg).
    
    Raises:
        ValueError: If video and PPG cannot be aligned.
    """
    pass
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=rppglib --cov-report=html

# Run only failed tests
pytest --last-failed
```

### Writing Tests

- Use `pytest` framework
- Test both positive and negative cases
- Include edge cases
- Keep tests fast and isolated

```python
import numpy as np
import pytest
from rppglib.face_utils import detect_face

def test_detect_face():
    """Test face detection on sample image."""
    # Load test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # This should not raise an error
    landmarks = detect_face(test_image)
    
    # Check output shape
    if landmarks is not None:
        assert landmarks.shape == (68, 2)

def test_detect_face_no_face():
    """Test face detection on image without face."""
    # Create image without face
    blank_image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Should return None
    landmarks = detect_face(blank_image)
    assert landmarks is None
```

## 📚 Documentation

### Updating Documentation

1. **Keep it current** - Update docs with code changes
2. **Clear and concise** - Avoid unnecessary details
3. **Include examples** - Show how to use features
4. **Use consistent formatting** - Follow existing style

### Documentation Structure

```
mcd_rppg/
├── README.md              # Main documentation
├── DATASET.md             # Dataset documentation
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # License information
└── preprocessing/
    └── README.md          # Preprocessing documentation
```

### Building Documentation

```bash
# Install mkdocs
pip install mkdocs mkdocs-material

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
```

## 🎁 Recognition

All contributors will be recognized in the project's contributors list. Significant contributions may be invited to become maintainers.

## 📞 Support

For questions about contributing:

1. **GitHub Discussions** - For general questions
2. **GitHub Issues** - For bug reports and feature requests
3. **Pull Requests** - For code contributions

---

**Thank you for contributing to MCD-rPPG!**

Your contributions help advance remote photoplethysmography research and make healthcare more accessible.
