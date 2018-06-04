**********
bifrost.py
**********
bifrost.py is a set of tools to read and interact with output from Bifrost simulations.

Installation
============
bifrost.py is part of the helita library. Helita documentation can be found here:

    http://helita.readthedocs.io/en/latest/index.html

This section is taken directly from the link above.

Pre-requisites
--------------
To make use of helita you need a Fortran compiler (GFortran is recommended), because some modules are compiled from Fortran. In addition, before attempting to install helita you need the following:

* `Python (2.7.x, 3.4.x or later) <https://www.python.org>`__
* `Astropy <http://www.astropy.org>`__
* `NumPy <http://www.numpy.org>`__
* `SciPy <https://www.scipy.org>`__

In addition, the following packages are also recommended to take advantage of all the features:

* `Matplotlib <https://matplotlib.org>`__
* `h5py <http://www.h5py.org>`__
* `netCDF4 <https://unidata.github.io/netcdf4-python/>`__
* `Cython <http://cython.org>`__
* `pandas <http://pandas.pydata.org>`__
* `beautifulsoup4 <https://www.crummy.com/software/BeautifulSoup/>`__

Helita will install without the above packages, but functionality will be limited.

All of the above packages are available through Anaconda, and that is the recommended way of setting up your Python distribution.

Cloning git repository
----------------------
The easiest way is to use git to clone the repository. To grab the latest version of helita and install it::

	$ git clone https://github.com/ITA-solar/helita.git
	$ cd helita
	$ python setup.py install
  

Non-root install
----------------
If you don't have write permission to your Python packages directory, use the following option with setup.py::

	$ python setup.py install --user

This will install helita under your home directory (typically ~/.local)

Developer install
-----------------
If you want to install helita but also actively change the code or contribute to its development, it is recommended that you do a developer install instead::

	$ python setup.py develop

This will set up the package, such as the source files used, from the git repository that you cloned (only a link to it is placed on the Python packages directory). Can also be combined with the *-user* flag for local installs::

	$ python setup.py develop --user

Installing with different C or Fortran compilers
------------------------------------------------
The procedure above will compile the C and Fortran modules using the default gcc/gfortran compilers. It will fail if these are not available in the system. If you want to use a different compiler, please use *setup.py* with the *-compiler=xxx* and/or *-fcompiler=yyy* options, where *xxx*, *yyy* are C and Fortran compiler families (names depend on system). To check which Fortran compilers are available in your system, you can run::

	$ python setup.py build --help-fcompiler

and to check which C compilers are available::

	$ python setup.py build --help-compiler

BifrostData Class
=================

bifrost.py includes the BifrostData class (among others). The following allows to create the class for snapshot number 430 from simulation 'cb10f' from directory '/net/opal/Volumes/Amnesia/mpi3drun/Granflux'::

	from helita.sim import bifrost as br
	dd = br.BifrostData('cb10f', snap = 430, fdir = '/net/opal/Volumes/Amnesia/mpi3drun/Granflux')

The snapshot(s) being read can be defined when creating the object, or set/changed anytime later. Snaps can be ints, arrays, or lists. 

Getting Variables & Quantities
==============================

get_var and get_varTime can be used to read variables as well as to calculate quantities (with a call to _get_quantity). iix, iiy, and iiz can be specified to slice the return array (they can be ints, arrays, or lists). If no snapshot is specified, the current snap value will be used (either the initialized one, which is the first in the series, or the most recent snap used in set_snap or in a call to get_var). To get variable 'r' at snap 430 with only values at y = 200, z = 5, and z = 7::

	var1 = dd.get_var('r', snap = 430, iiy = 200, iiz = [5, 7])

get_varTime can be used in the same fashion, with the added option of reading a specific variable or quantity from many snapshots at once. Its return arrays have an added dimension for time.