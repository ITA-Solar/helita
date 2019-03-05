# -*- coding: utf-8 -*-
"""
Tests for the rh15d module
"""

import numpy as np
from helita.sim import rh15d

TMP_ATOM_FILENAME = 'atom.tmp'
TEST_LEVELS = [
    "     0.000   2.00    'CA II 3P6 4S 2SE    '    1         0",
    " 13650.190   4.00    'CA II 3P6 3D 2DE 3  '    1         1",
    " 13710.880   6.00    'CA II 3P6 3D 2DE 5  '    1         2",
    " 25191.510   2.00    'CA II 3P6 4P 2PO 1  '    1         3",
    " 25414.400   4.00    'CA II 3P6 4P 2PO 3  '    1         4",
    " 95785.470   1.00    'CA III 3P6 1SE      '    2         5"
]

TEST_LEVELS_DATA = np.array([(    0.  , 2., 'CA II 3P6 4S 2SE', 1, 0),
                             (13650.19, 4., 'CA II 3P6 3D 2DE 3', 1, 1),
                             (13710.88, 6., 'CA II 3P6 3D 2DE 5', 1, 2),
                             (25191.51, 2., 'CA II 3P6 4P 2PO 1', 1, 3),
                             (25414.4 , 4., 'CA II 3P6 4P 2PO 3', 1, 4),
                             (95785.47, 1., 'CA III 3P6 1SE', 2, 5)],
                            dtype=[('energy', '<f8'), ('g_factor', '<f8'),
                                   ('label', '<U30'), ('stage', '<i4'),
                                   ('level_no', '<i4')])

TEST_ATOM_RH = """# Calcium II: 5 levels + continuum
  CA
# Nlevel  Nline   Ncont   Nfixed
    2       2       1        0
#  E[cm^-1]    g           label[20]         stage    levelNo
#                     |----|----|----|----
     0.000   2.00    'CA II 3P6 4S 2SE    '    1         0
 13650.190   4.00    'CA II 3P6 3D 2DE 3  '    1         1
 3 0 3.412E-01 PRD  100 ASYMM 30.0 450.0 BARKLEM 234. 0.223 1.00 0.00 1.48E08 1.0E-00
 4 0 6.807E-01 VOIGT 50 ASYMM 10.0 450.0 BARKLEM 234. 0.223 1.00 0.00 1.50E08 1.0E-00
#   CA II 3P6 4S 2SE
  5   0   2.0363E-23      15        EXPLICIT        35.0
  104.4   2.0363E-23
  100.0   2.0974E-23
   95.0   2.1455E-23
   90.0   2.1704E-23
   85.0   2.1715E-23
   80.0   2.1489E-23
   75.0   2.1025E-23
   70.0   2.0332E-23
   65.0   1.9419E-23
   60.0   1.8302E-23
   55.0   1.7001E-23
   50.0   1.5539E-23
   45.0   1.3944E-23
   40.0   1.2248E-23
   35.0   1.0486E-23
 TEMP    6          3000.0     5000.0     7000.0    15000.0    50000.0   100000.0
 OMEGA   1  0    2.378E+00  2.284E+00  2.203E+00  1.920E+00  1.961E+00  1.846E+00
 OMEGA   2  0    3.568E+00  3.426E+00  3.304E+00  2.879E+00  2.942E+00  2.770E+00
 TEMP    6          3000.0     5000.0     7000.0    15000.0    50000.0   100000.0
 CI      0  5    4.580E-18  4.580E-18  4.580E-18  4.580E-18  4.580E-18  4.580E-18
SUMMERS  1.0
SHULL82 0 3 0.00e+00 1.31e+05 4.70e-13 6.24e-01 0.00e-03 4.42e-02 1.57e+05 3.74e+05
AR85-CDI   0   3   2
    11.30     3.60    -9.60     7.20    -9.06
    16.60    14.58    -4.68     1.50   -14.40
AR85-CEA   0   3  6.00e-01

END
"""

def test_read_atom_levels():
    assert np.array_equal(rh15d.AtomFile.read_atom_levels(TEST_LEVELS, format='RH'),
                          TEST_LEVELS_DATA)

def test_AtomFile():
    temp_file = open(TMP_ATOM_FILENAME, 'w')
    temp_file.write(TEST_ATOM_RH)
    temp_file.close()

    data = rh15d.AtomFile(TMP_ATOM_FILENAME)
    assert np.array_equal(data.levels, TEST_LEVELS_DATA[:2])
    assert len(data.continua) == 1
    assert data.lines.shape == (2,)
    assert np.array_equal(data.lines['f_value'], np.array([0.3412, 0.6807]))
    assert data.continua[0]['cross_section'].shape == (15, 2)
    assert len(data.collision_temperatures) == 2
    assert len(data.collision_tables) == 7
    assert np.array_equal(data.collision_tables[0]['data'],
                          np.array([2.378, 2.284, 2.203, 1.92, 1.961, 1.846]))
    assert data.collision_tables[-1]['type'] == 'AR85-CEA'
