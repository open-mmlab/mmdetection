yapf -r -i --style .style.yapf mmdet/ configs/ tests/ tools/
isort -rc mmdet/ configs/ tests/ tools/
flake8 .
