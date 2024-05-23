# Installation

Clone the repository

```
git clone 
```

Then the submodules needed to be installed.

```
git submodule init
git submodule update --init --recursive
```

The `--init --update` is needed, because the submodule executorch also holds some submodules.