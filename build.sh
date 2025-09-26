rm -rf dist/*
python -m build
python -m twine upload dist/*
rm -rf dist/ build/ *.egg-info/
