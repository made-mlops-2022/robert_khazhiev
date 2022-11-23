.PHONY: up test coverage

up:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000
test:
	coverage run -m pytest test.py
coverage:
	coverage report
