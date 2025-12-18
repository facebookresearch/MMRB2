# Contributing to Multimodal RewardBench 2

We welcome contributions to MMRB2! This document provides guidelines for contributing.

## Ways to Contribute

- **Bug Reports**: Report issues with the benchmark build process or data
- **Documentation**: Improve README, add examples, fix typos
- **Source Modules**: Add support for new benchmark sources
- **Evaluation**: Contribute evaluation scripts and baselines

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/MMRB2.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions
- Include comments for complex logic

## Pull Request Process

1. Ensure your code passes any existing tests
2. Update documentation if needed
3. Describe your changes in the PR description
4. Link any related issues

## Adding New Source Modules

To add support for a new benchmark source:

1. Create a new file in `benchmark/sources/` (e.g., `new_source.py`)
2. Implement the required functions:
   - `download(data_dir, force=False)`: Download source data
   - `load_prompts(data_dir, required_keys=None)`: Load and return prompts
3. Register the source in `benchmark/sources/__init__.py`
4. Test the module thoroughly
5. Submit a PR with documentation

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the 
CC BY-NC 4.0 License.

## Questions?

Open an issue for questions or reach out to the maintainers.

Thank you for contributing!

