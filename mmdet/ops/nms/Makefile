PYTHON=${PYTHON:-python}

all:
	echo "Compiling nms kernels..."
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -f *.so
