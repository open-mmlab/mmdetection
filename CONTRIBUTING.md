# Contributing to mmdetection

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest mmdetection
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to mmdetection,
please contact Kai Chen (chenkaidev[at]gmail[dot]com). We will much appreciate your contribution.

## Code style

### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.
We use [flake8](http://flake8.pycqa.org/en/latest/) as the linter and [yapf](https://github.com/google/yapf) as the formatter.
Please upgrade to the latest yapf (>=0.27.0) and refer to the [configuration](.style.yapf).

>Before you create a PR, make sure that your code lints and is formatted by yapf.

### C++ and CUDA
We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).