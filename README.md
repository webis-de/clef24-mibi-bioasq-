# 🏥 mibi-bioasq

MiBi at BioASQ 2024

## Installation

1. Install [Python 3.10](https://python.org/downloads/).
2. Create and activate a virtual environment:

    ```shell
    python3.10 -m venv venv/
    source venv/bin/activate
    ```

3. Install project dependencies:

    ```shell
    pip install -e .
    ```

## Usage

Run the CLI with:

```shell
mibi --help
```

For example, to answer the questions from the file `data/training12b_new.json` and then save the answered questions to `data/training12b_new_answered.json`, simply run:

```shell
mibi run data/training12b_new.json data/training12b_new_answered.json
```

## Development

Refer to the general [installation instructions](#installation) to set up the development environment and install the dependencies.
Then, also install the test dependencies:

```shell
pip install -e .[tests]
```

After having implemented a new feature, please check the code format, inspect common LINT errors, and run all unit tests with the following commands:

```shell
ruff check .                   # Code format and LINT
mypy .                         # Static typing
bandit -c pyproject.toml -r .  # Security
pytest .                       # Unit tests
```

## Contribute

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/webis-de/archive-query-log/issues). We also gratefully accept [pull requests](https://github.com/webis-de/archive-query-log/pulls)!

If you are unsure about anything, post an [issue](https://github.com/webis-de/archive-query-log/issues/new) or contact us:

- [heinrich.reimer@uni-jena.de](mailto:heinrich.reimer@uni-jena.de)
- [alexander.bondarenko@uni-jena.de](mailto:alexander.bondarenko@uni-jena.de)
- [adrian.viehweger@medizin.uni-leipzig.de](mailto:adrian.viehweger@medizin.uni-leipzig.de)
- [matthias.hagen@uni-jena.de](mailto:matthias.hagen@uni-jena.de)

We are happy to help!

## License

This repository is released under the [MIT license](LICENSE).
Files in the `data/` directory are exempt from this license.
