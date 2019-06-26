# Example usage

## Loading Bifrost snapshots

The `bifrost` module under `helita.sim` reads and manipulates the native output format from Bifrost simulations (not FITS). To load some data, do:

``` python
from helita.sim import bifrost
data = bifrost.BifrostData("cb24bih", snap=300, fdir='/data/cb24bih')
```

The first argument to `BifrostData()` is the basename for file names (an underscore, the snapshot number, and file extensions will be added, e.g. `cb24bih_300.snap`). The object `data` contains the data for a single snapshot, but when initialised no data is loaded into memory. *Simple variables* are available as a memmap. These are variables that do not require any transformation before being read (typically scalar variables at cell centres and vector variables at cell faces). A list of such variable names is kept at `data.simple_vars`, and can be accessed as `data.variables[myvar]` or `data.myvar`, e.g.:

``` python
>>> data.tg.shape
(504, 504, 496)
>>> data.variables['tg'] is data.tg
True
```

The method `get_var()` can be used to read or extract any variable, including *composite variables*, which typically combine several *simple variables* and involve shifting variables from cell centres to cell faces or vice-versa. For example, to obtain velocities in the z direction:

``` python
vz = data.get_var('uz')
```

By default, this velocity will be defined at cell faces. To instead obtain it at the cell centres, we need to manually shift the momentum in the z direction (with the `cstagger` utilities) and divide it by the density, e.g.:

``` python
from helita.sim import cstagger
rdt = data.r.dtype
cstagger.init_stagger(data.nz, data.dx, data.dy, data.z.astype(rdt),
                      data.zdn.astype(rdt), data.dzidzup.astype(rdt),
                      data.dzidzdn.astype(rdt))
vz_cell_centre = cstagger.zup(data.pz) / data.r
```


## Converting Bifrost to RH 1.5D and Multi3D atmospheres

The `BifrostData` object has methods to convert the output into different formats. For example, to save the data from a single snapshot to an input atmosphere for RH 1.5D:

``` python
data = bifrost.BifrostData("cb24bih", snap=300, fdir='/data/cb24bih')
data.write_rh15d('myfile.hdf5', desc='Some description')
```

The interface also allows writing only part of the atmosphere using python `slice` objects in the x, y, and z dimensions. For example, to write every second point in the x and y directions, and the first 200 points in the z direction, you would do:

``` python
sx = sy = slice(None, None, 2)
sz = slice(0, 200)
data.write_rh15d('myfile.hdf5', desc='Some description', sx=sx, sy=sy, sz=sz)
```

An RH 1.5D atmosphere file can have multiple snapshots. To write additional snapshots to a file using the `write_rh15d()` method, you need to manually load a new snapshot and then use the `append=True` option (it can still be `True` for the first write, as long as the output file doesn't exist already). For example, writing snapshots from 100 to 150 to a single file:

``` python
data = bifrost.BifrostData("cb24bih", snap=100, fdir='/data/cb24bih')
for i in range(100, 150):
    data.set_snap(i)
    data.write_rh15d('myfile_s100-150.hdf5', desc='Some description',
                     append=True)
```

Writing snapshots in Multi3D format is done similarly, but only one snapshot can be written by file. E.g.:

``` python
data.write_multi3d('myfile.atmos3d', desc='Some description')
```

The Multi3D interface also writes a mesh file (default name is `mesh.dat`).

### Bifrost 2D to RH 1.5D

RH 1.5D atmosphere files are always 3D, but the x and y dimensions can have a single element, meaning 1D and 2D geometries are possible. When converting 2D models from Bifrost to RH 1.5D, the resulting atmosphere will therefore have `ny = 1`. However, when converting many 2D snapshots to RH 1.5D it is advantageous to save the temporal dimension as a y dimension, because RH 1.5D is not parallel in the temporal dimension. There is a function called `bifrost2d_to_rh15d()` in `helita.sim.bifrost` that does this. For example:

``` python
import numpy as np
snaps = np.arange(100, 500)
bifrost.bifrost2d_to_rh15d(snaps, 'myfile2D.hdf5', 'simroot', 'mesh.dat',
                           '/path/to/simulation/', desc='Some description')
```

The resulting atmosphere variables will therefore have a shape of `(1, nx, nt, nz)` instead of `(nt, nx, 1, nz)`.
