rm -rf dist
#!/bin/bash
echo "start clean"

rm -rf _skbuild
rm -rf _line_profiler.c
rm -rf *.so
rm -rf build
rm -rf *.egg-info
rm -rf .eggs
rm -rf dist
rm -rf mb_work
rm -rf wheelhouse

rm distutils.errors || echo "skip rm"

CLEAN_PYTHON='find . -regex ".*\(__pycache__\|\.py[co]\)" -delete || find . -iname *.pyc -delete || find . -iname *.pyo -delete'
bash -c "$CLEAN_PYTHON"

# Remove inplace shared object files
find . -regex ".*so" -delete

echo "finish clean"
