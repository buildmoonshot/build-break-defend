.PHONY: setup test

setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

test:
	.venv/Scripts/python -m pytest
