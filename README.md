# dplPy
The Dendrochronology Program Library for Python

## Issues

We're using [ZenHub](https://app.zenhub.com/workspaces/opendendro-60ec698d8790d700171ceee8/board?repos=385244315) to manage our [GitHub Issues](https://github.com/opendendro/dplpy/issues)

## Builds

!!! Warning
    Prior to creating an enviroment, ensure that you are outside of `base` by doing `conda deactivate`.

Create a conda environment with python version 3 as default:

```
conda create -n dplpy3 python=3
```

When prompted for permission to install required packages (with `y/n`), select `y`. Upon finishing installing the required packages, reload your terminal (close and re-open terminal).

!!! Warning
    If terminal shows you are in `base`, exit with `conda deactivate`.

Activate your environment:

```
conda activate dplpy3
```

From within your environment, install [CSAPS](https://pypi.org/project/csaps/#description):

```
pip install -U csaps
```

Update your environment:

```
conda env update -f environment.yml --prune
```

We're using [ZenHub](https://zenhub.com) to manage our GitHub Issues
