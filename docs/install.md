# Installation

Helita needs **Python 3.4.x** or higher, use of [Python 2.x is discouraged](https://python3statement.org/). Helita relies on several standard scientific python libraries. Here is a list of required and optional (not all functions will work) packages:


| Required                                   | Optional                        |
|--------------------------------------------|---------------------------------|
| [astropy](http://astropy.org)              | [cython](https://cython.org/)   |
| [NumPy](http://numpy.scipy.org/)           | [numba](http://numba.pydata.org/) |
| [SciPy](http://www.scipy.org/)             | [bqplot](https://github.com/bloomberg/bqplot) |
| [xarray](http://xarray.pydata.org/)        | [ipywidgets](https://ipywidgets.readthedocs.io) |
| [h5py](https://www.h5py.org/)              | [matplotlib](https://matplotlib.org/) |
|                                            | [tqdm](https://tqdm.github.io/) |
|                                            | [specutils](https://specutils.readthedocs.io) |

Many of these should be installed when [astropy](http://astropy.org) is installed. All of the above Python packages are available through [Anaconda](https://docs.continuum.io/anaconda/), and that is the recommended way of setting up your Python distribution.

!!! warning "Compiled modules"
    Some helita modules are written in C and Fortran. To use them, you need to have respective compilers available in your system **before** you try to install helita (see below for help with compilers). The Fortran modules are optional, and most of helita works without them, but the C modules are not. **You need a C compiler to install helita!**

## Install from source

To install helita you need to clone the [repository](https://github.com/ITA-Solar/helita) (or download a [zip version](https://github.com/ITA-Solar/helita/archive/master.zip) if you don't have git) and then install with python:

```
git clone https://github.com/ITA-solar/helita.git
cd helita
python setup.py install
```

## Non-root install

If you don't have write permission to your Python packages directory, use the following option with `setup.py`:

    python setup.py install --user

This will install helita under your home directory (typically `~/.local`).

## Developer install

If you want to install helita but also actively change the code or contribute to its development, it is recommended that you do a developer install instead:

    python setup.py develop

This will set up the package such as the source files used are from the git repository that you cloned (only a link to it is placed on the Python packages directory). Can also be combined with the `--user` flag for local installs.

## Installing with different C or Fortran compilers

The procedure above will compile the C and Fortran modules using the default gcc/gfortran compilers. It will fail if at least a C compiler is not available in the system. If you want to use a different compiler, please use `setup.py` with the  `--compiler=xxx` and/or `--fcompiler=yyy` options, where `xxx`, `yyy` are C and Fortran compiler families (names depend on system). To check which Fortran compilers are available in your system, you can run:

    python setup.py build --help-fcompiler

and to check which C compilers are available:

    python setup.py build --help-compiler
