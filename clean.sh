rm -rf build
rm -rf dist
rm -rf *.egg-info
rm -rf pysparta/version.py
rm -rf pysparta.egg-info
rm -rf .pytest_cache
rm -f MANIFEST
find . -name "__pycache__" -print0 | xargs -0 -I {} /bin/rm -rf "{}"
