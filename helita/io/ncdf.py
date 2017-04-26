"""
NetCDF helper functions
"""

import os
import struct
import netCDF4 as nc


def ncdf_info(filename):
    """
    Reads a NetCDF file and displays information about its content.
    20130508-- Tiago.
    """
    def _attr_string(s):
        if type(s) in [type(''), type('')]:
            return '"%s"' % s
        else:
            return str(s)

    def _group_print(g, ind=0):
        gatname = ['global', 'group'][ind != 0]
        gsp = '    ' * ind
        if len(g.dimensions) > 0:
            print(('%sdimensions:' % gsp))
            for a, v in list(g.dimensions.items()):
                print(('\t%s = %i' % (a, len(v))))
        if len(g.variables) > 0:
            print(('%svariables:' % gsp))
            for k, v in list(g.variables.items()):
                dd = str('(' + ', '.join(v.dimensions) + ')')
                print(('%s\t%s %s%s' % (gsp, str(v.dtype).ljust(10), k, dd)))
                for a, av in list(v.__dict__.items()):
                    print(('%s\t%s%s = %s' % (gsp, ' ' * 15, a,
                                              _attr_string(av))))
        if len(g.__dict__) > 0:
            print(('%s%s attributes:' % (gsp, gatname)))
            for a, av in list(g.__dict__.items()):
                print(('%s\t%s = %s' % (gsp, a, _attr_string(av))))

    try:
        f = nc.Dataset(filename, 'r')
    except IOError as e:
        print(e)
        return
    print(filename)
    _group_print(f, ind=0)
    if len(f.groups) > 0:
        for k, g in list(f.groups.items()):
            print(('group: %s {' % k))
            _group_print(g, ind=1)
            if len(g.groups) > 0:
                for ki, gi in list(g.groups.items()):
                    print(('    group: %s {' % ki))
                    _group_print(gi, ind=2)
                    if len(gi.groups) > 0:
                        for kii, gii in list(gi.groups.items()):
                            print(('        group: %s {' % kii))
                            _group_print(gii, ind=3)
                            print(('        } // group %s' % kii))
                    print(('    } // group %s' % ki))
            print(('} // group %s' % k))
    f.close()
    return


def copy_ncdf(filein, transp=[], remove=[], memGb=16, step=None, tlimit=None,
              fileout=None):
    ''' Copies variables from a netcdf file and writes them in a new file.

    IN:

    filein: input netCDF file
    transp: list of strings with variable names. These variables will be transposed.
    remove: list of strings, variables in this list will not be copied.
    memGb: integer, size of RAM before which transpose is made in memory
    '''

    if fileout is None:
        fileout = os.path.splitext(
            filein)[0] + '_copy' + os.path.splitext(filein)[1]
    if os.path.isfile(fileout):
        raise IOError(("%s already exists"
                       ", remove or rename for copy to proceeed." % fileout))
    finp = nc.Dataset(filein, 'r')
    fout = nc.Dataset(fileout, 'w')
    # copy dimensions
    for d in list(finp.dimensions.keys()):
        if tlimit and d == 'nt':
            fout.createDimension(d, tlimit)
        else:
            fout.createDimension(d, len(finp.dimensions[d]))
    # copy attributes
    for d in list(finp.__dict__.keys()):
        if tlimit and d == 'ntnum':
            setattr(fout, d, finp.__dict__[d][:tlimit])
        else:
            setattr(fout, d, finp.__dict__[d])
    # create variables
    for vname, v in list(finp.variables.items()):
        if vname in transp:
            print(('Transposing ' + vname))
            myvar = fout.createVariable(vname, v.dtype, v.dimensions[::-1])
            # size of variable in Gb
            var_size = np.prod(v.shape) * struct.calcsize(v.dtype.str) / 2.**30
            if var_size < memGb:
                buf = v[:]
                myvar[:] = buf.T
            else:
                # transpose: fit as much as possible in memory to minimize I/O
                if step is None:
                    step = int(memGb / (np.prod(v.shape[1:]) *
                               struct.calcsize(v.dtype.str) / 2.**30))
                for i in range(0, v.shape[0], step):
                    print(i, step, v.shape[0])
                    buf = v[i:i + step]
                    myvar[..., i:i + buf.shape[0]] = buf.T
        elif vname in remove:
            print(('Skipping ' + vname))
            continue
        else:
            print(('Copying ' + vname))
            if tlimit:
                myvar = fout.createVariable(vname, v.dtype, v.dimensions)
                myvar[:] = v[:tlimit]
            else:
                myvar = fout.createVariable(vname, v.dtype, v.dimensions)
                myvar[:] = v[:]
        # copy variable attributes
        for d in list(v.__dict__.keys()):
            setattr(myvar, d, v.__dict__[d])
    # close files
    finp.close()
    fout.close()
    return


def copy_var(filein, fileout, vars=[], step=15):
    ''' Copies variables from one netcdf file to another.
    IN:

    filein: input netCDF file, where the variable(s) will be read
    fileout: output netCDF file (must exist), where to copy variable(s)
    vars: list of strings with variable names to be copied.
    '''
    finp = nc.Dataset(filein, 'r')
    fout = nc.Dataset(fileout, 'a')
    for v in vars:
        print(v)
        fv = finp.variables[v]
        dd = fv.dimensions
        for d in dd:
            if d not in fout.dimensions:
                print('(WWW) copy_var: dimension ' +
                      '%s not found in %s, creating it.' % (d, fileout))
                fout.createDimension(d, len(finp.dimensions[d]))
            if len(finp.dimensions[d]) != len(fout.dimensions[d]):
                raise ValueError('copy_var: dimension has size' +
                    '%i in %s and %i in %s. Aborting'% (len(finp.dimensions[d]),
                    filein, len(fout.dimensions[d], fileout)))
        if v not in fout.variables:
            myvar = fout.createVariable(v, fv.dtype, fv.dimensions)
        else:
            myvar = fout.variables[v]
        # copy variable
        for i in range(0, fv.shape[0], step):
            print(i, i + step)
            buf = finp.variables[v][i:i + step]
            myvar[i:i + buf.shape[0]] = buf
        # copy variable attributes
        for d in list(fv.__dict__.keys()):
            buf = v[i:i + step]
            setattr(myvar, d, fv.__dict__[d])
    # close files
    finp.close()
    fout.close()
    return


def getvar(infile, var, group=False, memmap=False):
    ''' Reads a variable from a NetCDF file.

    IN:
    file (string): NetCDF filename
    var (string) : variable name
    memmap (bool): [optional] if True, will return the variable object
                  (not in memory), instead of reading it into memory.
    OUT:
    result (array) : array with the requested variable

    --Tiago, 20090629
    '''
    f = nc.Dataset(infile, mode='r')
    ds = f
    if group:
        ds = f.groups[group]
    if var not in ds.variables:
        raise KeyError('getvar: variable %s not in %s' % (var, infile))
    if not memmap:
        result = np.array(ds.variables[var][:])
        f.close()
    else:
        result = ds.variables[var]
    return result


def updatevar(infile, var, data, group=False):
    ''' Updates a variable in a NetCDF file.

    IN:
    file (string): NetCDF filename
    var (string) : variable name
    data : array with data to overwrite. Must be same dimensions as in the file.

    --Tiago, 20111216
    '''
    f = nc.Dataset(file, mode='a')
    ds = f
    if group:
        ds = f.groups[group]
    if var not in ds.variables:
        raise KeyError('getvar: variable %s not in %s' % (var, file))
    ds.variables[var][:] = data[:]
    f.close()
    return


def merge_snaps(origf, newf, order=False, unique=True):
    ''' Merges two line profile files (lte.x/multi3d ncdf format) into the
        first line profile file.

    IN:
    origf, newf: ncdf filenames.
    order : if True, sorts all the variables (and ntnum) according to ntnum.
    unique: if True, eliminates duplicate snapshots in case both files have
            some common snapshots.

    OUT:
    None. (Results saved in orig)

    --Tiago, 20080128
    '''
    new = nc.Dataset(newf, mode='r')
    nt_orig = np.array(getattr(orig, 'ntnum')).astype('i')
    nt_new = np.array(getattr(new, 'ntnum')).astype('i')
    # for netcdf4 having an extra dim
    if not nt_new.shape:
        nt_new = np.array([nt_new])
    if not nt_orig.shape:
        nt_orig = np.array([nt_orig])
    # select nt's not already in the file
    if unique:
        idx2 = []
        for i in range(len(nt_new)):
            if nt_new[i] not in nt_orig:
                idx2.append(i)
        if idx2 == []:
            print('*** All snapshots common, not merging.')
            return
        idx2 = np.array(idx2).astype('i')
    else:
        idx2 = np.arange(len(nt_new))
    ntnum = np.concatenate((nt_orig, nt_new[idx2]))
    # sort by nt number?
    if order:
        idx = np.argsort(ntnum)
    else:
        idx = np.arange(len(ntnum))
    print('--- Merging...')
    # ending snapshot number for orig file
    ent = orig.variables['prof_int'].shape[0]
    # merge variables in first (unlimited) dimension
    for v in orig.variables:
        nvs = new.variables[v].shape
        if ncdf4:
            nvar = new.variables[v]
        else:
            nvar = np.array(new.variables[v][:])
        # must put if for ncdf3 to put new array in memory and fancy index it
        print(v)  # , ovs,nvs
        # fix case onf only one nt
        if len(nt_new) == 1:
            orig.variables[v][ent] = nvar[0]
        else:
            orig.variables[v][ent:] = nvar[idx2]
        if order:
            orig.variables[v][:] = orig.variables[v][idx]
    # update ntnum
    setattr(orig, 'ntnum', ntnum[idx])
    orig.close()
    new.close()
    print('--- Successfully merged %s into %s.' % (newf, origf))
    return
