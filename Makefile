.PHONY: readme test help install

help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

test: ## run pytest suite
	poetry run pytest

readme: ## render Quarto readme
	poetry run quarto render README.qmd
	sed -i '/<div><style>/,/<\/style>/d' README.md
	sed -i '/<div><style>/,/<\/style>/d' README.md
	sed -i '/<\/div>/d' README.md

install: ## install in poetry venv
	poetry install