# Contributing to Ternary-Zero

Thank you for your interest in contributing to Ternary-Zero! We welcome contributions from the community to help improve this project.

## How to Contribute

There are many ways to contribute, including:

- Reporting bugs and issues
- Suggesting new features
- Improving documentation
- Writing tutorials or examples
- Submitting code fixes and new features
- Reviewing pull requests

## Reporting Issues

Before submitting an issue, please check if it has already been reported. When reporting a bug, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Information about your environment (OS, Python version, CUDA version, GPU model)
- Any relevant logs or error messages
- Minimal reproducible example if possible

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code follows the project's coding standards.
5. Update the documentation as needed.
6. Submit your pull request with a clear description of changes.

### Coding Standards

- Follow the existing code style in the project.
- Write clear, meaningful commit messages.
- Comment your code where necessary, especially for complex logic.
- Add type hints for Python functions where applicable.
- For Rust code, follow the Rust API guidelines.

### Testing

- Unit tests should be added for new functionality.
- Run the full test suite before submitting: `cargo test` and `python -m pytest`
- Benchmarks should be updated if performance is affected.

## Development Setup

### Prerequisites

- Python 3.9+
- Rust toolchain (via rustup)
- CUDA Toolkit 12.x (for GPU development)
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/ternary-zero.git
cd ternary-zero

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -e .[dev]
pip install maturin

# Build the Rust extension in development mode
maturin develop --release
```

### Running Tests

```bash
# Run Rust unit tests
cargo test

# Run Python tests
python -m pytest tests/

# Run benchmarks
cargo bench --bench cpu_kernels
cargo bench --bench gpu_kernels  # Requires CUDA
```

## Code Review Process

All pull requests will be reviewed by at least one maintainer. The review process includes:

1. Checking for correctness and completeness
2. Ensuring adherence to coding standards
3. Verifying that tests pass and coverage is adequate
4. Evaluating performance implications
5. Providing feedback and requesting changes if needed

## Community

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions with the project.

## Getting Help

If you need help with your contribution, feel free to:

- Ask questions in the issue tracker
- Reach out to maintainers directly
- Look at existing issues and pull requests for guidance

Thank you again for contributing to Ternary-Zero!