"""
Test suite for multi3d.py
"""
import os
import pytest
import tarfile
import numpy as np
from shutil import rmtree
from pkg_resources import resource_filename
from helita.sim import multi3d

TEST_FILES = ['ie_+0.00_+0.00_+1.00_allnu', 'multi3d.input', 'out_atm',
              'out_nu', 'out_par', 'out_pop', 'out_rtq',
              'snu_+0.00_+0.00_+1.00_allnu']
TEST_TARBALL = resource_filename('helita', 'data/multi3d_output.tar.bz2')
TEST_DIR = resource_filename('helita', 'data/multi3d_test')

# A sample of keyword values from multi3d.input
INPUT_VALUES = {'atmosid': 'falc.5x5x82', 'atom': '../input/atoms/atom.h3',
                'convlim': 0.001, 'n_scratch': 10}

def unpack_data(source_tarball, files, output_directory):
    """Unpack test data to temporary directory."""
    assert os.path.isfile(source_tarball), 'Could not find test data files.'
    tarball = tarfile.open(source_tarball)
    for outfile in files:
        tarball.extract(outfile, path=output_directory)


def test_Multi3dOut():
    """
    Tests Multi3dOut class
    """
    unpack_data(TEST_TARBALL, TEST_FILES, TEST_DIR)
    data = multi3d.Multi3dOut(directory=TEST_DIR, printinfo=False)
    with pytest.raises(TypeError):
        data.set_transition(1, 0)
    data.readall()
    # From intput file
    for key in INPUT_VALUES:
        assert data.theinput[key] == INPUT_VALUES[key]
    # From out_par
    assert np.isclose(data.geometry.wmu, np.array([1.57079633, 1.57079633,
                                                   1.57079633, 1.57079633])).all()
    assert np.isclose(data.atom.g, np.array([2., 8., 18., 1.])).all()
    assert data.atom.crout == b'gencol'
    assert data.atom.nline == 3
    assert data.atom.nlevel == 4
    assert data.atom.ncont == 3
    assert np.array_equal(data.atom.ilin,
                          np.array([[0, 1, 2, 0], [1, 0, 3, 0],
                                    [2, 3, 0, 0], [0, 0, 0, 0]], dtype='i'))
    # From out_nu
    assert np.array_equal(data.outff, np.arange(1, 245))
    # From out_pop
    assert np.isclose(data.atom.ntot[0, 0, ::15],
                      np.array([1.04571361e+10, 4.12490015e+10,
                                1.16667064e+11, 3.66281941e+13,
                                2.30036022e+16, 1.25963806e+17])).all()
    # From out_rtq
    assert data.atmos.x500.sum() == 0
    assert np.isclose(data.atom.dopfac[-1, -1, 3::20],
                      np.array([1.2885583e-04, 5.1594885e-05,
                                3.6189285e-05, 3.0924344e-05])).all()
    # Line parameters
    data.set_transition(1, 0)
    assert np.isclose(data.d.l.value,
                      np.array([911.7, 850., 800., 750., 700., 600.])).all()
    # Emergent intensity
    data.set_transition(3, 2)
    ie = data.readvar('ie')
    assert np.array_equal(ie[0, 0], ie[-1, -1])
    assert np.isclose(ie[0,0,5::20],
                      np.array([2.9016188e-05, 1.1707955e-05, 3.8370090e-06,
                                4.9833211e-06, 1.8675400e-05])).all()


def test_clean():
    """Delete temporary directory and all its contents."""
    if os.path.isdir(TEST_DIR):
        rmtree(TEST_DIR)
