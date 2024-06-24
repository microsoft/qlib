# This is an experimental branch of "`FI`nancial `CO`pilot of `Qlib`"

## Installation

- To run this module, you need to first install Qlib following the instruction in [install-from-source](/README.md#install-from-source) or follow:

```python
python -m pip install git+https://github.com/microsoft/qlib.git@finco
```

- then you need to install other dependencies of finco:
```python
python -m pip install pydantic openai python-dotenv
```

## Quick run

To run this module, you can start the workflow easily with one command:

```sh
cd qlib/finco; python cli.py "your prompt"
```