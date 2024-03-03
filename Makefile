.PHONY: readme test help install docs

help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

test: install ## run pytest suite
	poetry run pytest

snapshot: ## snapshot test
	R CMD BATCH tests/r/run.R

readme: ## render Quarto readme
	poetry run quarto render docs/get_started.qmd --to gfm
	mv -f docs/get_started.md README.md

lint: ## run the lint checkers
	poetry run ruff format marginaleffects
	poetry run ruff format tests

install: ## install in poetry venv
	poetry install

docs: readme ## build docs
	poetry run mkdocs build