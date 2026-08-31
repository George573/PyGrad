.PHONY: style style-check

# Apply deterministic, safe code-style refactors.
style:
	python scripts/style.py

# Verify style without modifying files (suitable for CI).
style-check:
	python scripts/style.py --check
