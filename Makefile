.PHONY: style quality

check_dirs := klue_baseline/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	mypy --install-types --non-interactive $(check_dirs)
