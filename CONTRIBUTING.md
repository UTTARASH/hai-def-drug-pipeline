# Contributing to Agnosti-Path

Thank you for your interest in contributing to Agnosti-Path! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in [Issues](https://github.com/UTTARASH/hai-def-drug-pipeline/issues)
- If not, create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version, GPU)

### Suggesting Features
- Open an issue with the `enhancement` label
- Describe the feature and its use case
- Explain how it fits with the project goals

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and under 50 lines when possible

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pipeline --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py
```

## 📚 Documentation

- Update README.md if adding new features
- Add docstrings following Google style
- Update type hints for new functions

## 🎯 Areas for Contribution

### High Priority
- [ ] Add more disease targets
- [ ] Improve model caching
- [ ] Add batch processing
- [ ] Optimize memory usage

### Medium Priority
- [ ] Add visualization options
- [ ] Improve error handling
- [ ] Add progress bars
- [ ] Support for more file formats

### Documentation
- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Video tutorials
- [ ] Case studies

## 💬 Questions?

- Open a [Discussion](https://github.com/UTTARASH/hai-def-drug-pipeline/discussions)
- Join our Discord (coming soon)

Thank you for contributing! 🎉
