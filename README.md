# Helita

Helita is a Python library for solar physics focused on interfacing with code and projects from the [Institute of Theoretical Astrophysics](http://astro.uio.no) (ITA) at the [University of Oslo](https://www.uio.no). The name comes from Helios + ITA.

Currently, the library is a loose collection of different scripts and classes with varying degrees of portability and usefulness.

## Installation

To make use of helita **you need a Fortran compiler** ([GFortran](https://gcc.gnu.org/wiki/GFortran) is recommended), because some modules are compiled from Fortran. In addition, before attempting to install helita you need the following:

 * [Python](http://www.python.org) (2.7.x, 3.4.x or later)
 * [Astropy](http://astropy.org)
 * [NumPy](http://numpy.scipy.org/)
 * [SciPy](http://www.scipy.org/)

The following packages are also recommended to take advantage of all the features:

* [Matplotlib](http://matplotlib.sourceforge.net/) (1.1+)

* [netCDF4](https://unidata.github.io/netcdf4-python/)
* [Cython](http://www.cython.org)
* [pandas](http://pandas.pydata.org/)
* [beautifulsoup4](http://www.crummy.com/software/BeautifulSoup/)

Helita will install without the above packages, but functionality will be limited.

All of the above Python packages are available through [Anaconda](https://docs.continuum.io/anaconda/), and that is the recommended way of setting up your Python distribution.

Next, use git to grab the latest version of helita:

    git clone https://github.com/ITA-solar/helita.git
    cd helita
    python setup.py install

### Non-root install

If you don't have write permission to your Python packages directory, use the following option with `setup.py`:

    python setup.py install --user

This will install helita under your home directory (typically `~/.local`).

### Developer install

If you want to install helita but also actively change the code or contribute to its development, it is recommended that you do a developer install instead:

    python setup.py develop

This will set up the package such as the source files used are from the git repository that you cloned (only a link to it is placed on the Python packages directory). Can also be combined with the `--user` flag for local installs.


## Documentation

Some form of documentation will be made available at http://helita.readthedocs.org, but right now there is little documentation other than that in docstrings.
