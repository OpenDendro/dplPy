# dplPy
The Dendrochronology Program Library for Python

## Issues

We're using [ZenHub](https://app.zenhub.com/workspaces/opendendro-60ec698d8790d700171ceee8/board?repos=385244315) to manage our [GitHub Issues](https://github.com/opendendro/dplpy/issues)

## Builds

Create a conda environment:

```
conda env create -f environment.yml
```

Activate the environment:

```
conda activate dplpy
```

## Tests

From terminal:

```
$ python summary.tucson.py ./tests/data/filename.rwl
```

From Python3 console or notebook cell:

```
>>> import dplpy as dpl
>>> dpl.summary.tucson("./tests/data/filename.rwl")
```