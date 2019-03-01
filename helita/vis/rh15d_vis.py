"""
Set of programs and tools visualise the output from RH, 1.5D version
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pkg_resources import resource_filename
from ipywidgets import interact, fixed, Dropdown, IntSlider, FloatSlider
from scipy.integrate.quadrature import cumtrapz
from scipy.interpolate import interp1d
from astropy import units as u
from ..utils.utilsmath import planck, voigt


class Populations:
    """
    Class to visualise the populations from an RH 1.5D object.
    """
    def __init__(self, rh_object):
        self.rhobj = rh_object
        self.atoms = [a for a in dir(self.rhobj) if a[:5] == 'atom_']
        self.display()

    def display(self):
        """
        Displays a graphical widget to explore the level populations.
        Works in jupyter only.
        """
        atoms = {a.split('_')[1].title(): a for a in self.atoms}
        quants = ['Populations', 'LTE Populations', 'Departure coefficients']
        #nlevel = getattr(self.rhobj, self.atoms[0]).nlevel
        nx, ny, nz = self.rhobj.atmos.temperature.shape
        if nx == 1:
            x_slider = fixed(0)
        else:
            x_slider = (0, nx - 1)
        if ny == 1:
            y_slider = fixed(0)
        else:
            y_slider = (0, ny - 1)

        def _pop_plot(atom):
            """Starts population plot"""
            pop = getattr(self.rhobj, atom).populations
            height = self.rhobj.atmos.height_scale[0, 0] / 1e6  # in Mm
            _, ax = plt.subplots()
            pop_plot, = ax.plot(height, pop[0, 0, 0])
            ax.set_xlabel("Height (Mm)")
            ax.set_ylabel("Populations")
            ax.set_title("Level 1")
            return ax, pop_plot

        ax, p_plot = _pop_plot(self.atoms[0])

        @interact(atom=atoms, quantity=quants, y_log=False,
                  x=x_slider, y=y_slider)
        def _pop_update(atom, quantity, y_log=False, x=0, y=0):
            nlevel = getattr(self.rhobj, atom).nlevel

            # Atomic level singled out because nlevel depends on the atom
            @interact(level=(1, nlevel))
            def _pop_update_level(level=1):
                n = getattr(self.rhobj, atom).populations[level - 1, x, y]
                nstar = getattr(
                    self.rhobj, atom).populations_LTE[level - 1, x, y]
                if quantity == 'Departure coefficients':
                    tmp = n / nstar
                    ax.set_ylabel(quantity + ' (n / n*)')
                elif quantity == 'Populations':
                    tmp = n
                    ax.set_ylabel(quantity + ' (m$^{-3}$)')
                elif quantity == 'LTE Populations':
                    tmp = nstar
                    ax.set_ylabel(quantity + ' (m$^{-3}$)')
                p_plot.set_ydata(tmp)
                ax.relim()
                ax.autoscale_view(True, True, True)
                ax.set_title("Level %i, x=%i, y=%i" % (level, x, y))
                if y_log:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("linear")


class SourceFunction:
    """
    Class to visualise the source function and opacity from an RH 1.5D object.
    """
    def __init__(self, rh_object):
        self.rhobj = rh_object
        self.display()

    def display(self):
        """
        Displays a graphical widget to explore the source function.
        Works in jupyter only.
        """
        nx, ny, nz, nwave = self.rhobj.ray.source_function.shape
        if nx == 1:
            x_slider = fixed(0)
        else:
            x_slider = (0, nx - 1)
        if ny == 1:
            y_slider = fixed(0)
        else:
            y_slider = (0, ny - 1)
        tau_levels = [0.3, 1., 3.]
        ARROW = dict(facecolor='black', width=1., headwidth=5, headlength=6)
        #SCALES = ['Height', 'Optical depth']

        def __get_tau_levels(x, y, wave):
            """
            Calculates height where tau=0.3, 1., 3 for a given
            wavelength index.
            Returns height in Mm and closest indices of height array.
            """
            h = self.rhobj.atmos.height_scale[x, y].dropna('height')
            tau = cumtrapz(self.rhobj.ray.chi[x, y, :, wave].dropna('height'),
                           x=-h)
            tau = interp1d(tau, h[1:])(tau_levels)
            idx = np.around(interp1d(h,
                                     np.arange(h.shape[0]))(tau)).astype('i')
            return (tau / 1e6, idx)  # in Mm

        def _sf_plot():
            """Starts source function plot"""
            obj = self.rhobj
            sf = obj.ray.source_function[0, 0, :, 0].dropna('height')
            height = obj.atmos.height_scale[0, 0].dropna(
                'height') / 1e6  # in Mm
            bplanck = planck(obj.ray.wavelength_selected[0] * u.nm,
                             obj.atmos.temperature[0, 0].dropna('height') * u.K,
                             dist='frequency').value
            fig, ax = plt.subplots()
            ax.plot(height, sf, 'b-', label=r'S$_\mathrm{total}$', lw=1)
            ax.set_yscale('log')
            ax.plot(height, obj.ray.Jlambda[0, 0, :, 0].dropna('height'),
                    'y-', label='J', lw=1)
            ax.plot(height, bplanck, 'r--', label=r'B$_\mathrm{Planck}$',
                    lw=1)
            ax.set_xlabel("Height (Mm)")
            ax.set_ylabel(r"W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$")
            ax.set_title("%.3f nm" % obj.ray.wavelength_selected[0])
            lg = ax.legend(loc='upper center')
            lg.draw_frame(0)
            # tau annotations
            tau_v, h_idx = __get_tau_levels(0, 0, 0)
            for i, level in enumerate(tau_levels):
                xval = tau_v[i]
                yval = sf[h_idx[i]]
                ax.annotate(r'$\tau$=%s' % level,
                            xy=(xval, yval),
                            xytext=(xval, yval / (0.2 - 0.03 * i)),
                            arrowprops=ARROW, ha='center', va='top')
            return ax

        ax = _sf_plot()

        @interact(wavelength=(0, nwave - 1, 1), y_log=True,
                  x=x_slider, y=y_slider)
        def _sf_update(wavelength=0, y_log=True, x=0, y=0):
            obj = self.rhobj
            bplanck = planck(obj.ray.wavelength_selected[wavelength],
                             obj.atmos.temperature[x, y].dropna('height'),
                             units='Hz')
            quants = [obj.ray.source_function[x, y, :,
                                              wavelength].dropna('height'),
                      obj.ray.Jlambda[x, y, :, wavelength].dropna('height'),
                      bplanck]
            for i, q in enumerate(quants):
                ax.lines[i].set_ydata(q)
            ax.relim()
            ax.autoscale_view(True, True, True)
            ax.set_title("%.3f nm" % obj.ray.wavelength_selected[wavelength])
            # tau annotations:
            tau_v, h_idx = __get_tau_levels(x, y, wavelength)
            for i in range(len(tau_levels)):
                xval = tau_v[i]
                yval = quants[0][h_idx[i]]
                ax.texts[i].xy = (xval, yval)
                ax.texts[i].set_position((xval, yval / (0.2 - 0.03 * i)))
            if y_log:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")


class InputAtmosphere:
    def __init__(self, filename):
        self.atmos = xr.open_dataset(filename)
        self.filename = filename
        self.display()

    def display(self):
        """
        Displays a graphical widget to explore the input (HDF5) atmosphere.
        """
        ntsteps, nx, ny, nz = self.atmos.temperature.shape
        if ntsteps == 1:
            tslider = fixed(0)
        else:
            tslider = (0, ntsteps - 1)
        if nx == 1:
            x_slider = fixed(0)
        else:
            x_slider = (0, nx - 1)
        if ny == 1:
            y_slider = fixed(0)
        else:
            y_slider = (0, ny - 1)

        def _atmos_plot():
            """Starts source function plot"""
            EXCLUDES = ['x', 'y', 'z', 'snapshot_number']
            self.variables = [v for v in self.atmos.variables
                              if v not in EXCLUDES]
            nrows = int(np.ceil(len(self.variables) / 2.))
            fig, ax = plt.subplots(nrows, 2, sharex=True,
                                   figsize=(7, 2. * nrows))
            for i, v in enumerate(self.variables):
                var = self.atmos.variables[v]
                if v[:8].lower() == 'velocity':  # to km/s
                    ax.flat[i].plot(self.atmos.z[0] / 1e6, var[0, 0, 0] / 1.e3)
                    ax.flat[i].set_ylabel("%s (km/s)" % v.title())
                elif v.lower() == "hydrogen_populations":
                    ax.flat[i].plot(self.atmos.z[0] / 1e6,
                                    var[0, :, 0, 0].sum(axis=0))
                    ax.flat[i].set_ylabel("Htot (m^-3)")
                    ax.flat[i].set_yscale("log")
                else:
                    ax.flat[i].plot(self.atmos.z[0] / 1e6, var[0, 0, 0])
                    units = ''
                    if 'units' in var.attrs:
                        units = var.attrs['units']
                    ax.flat[i].set_ylabel("%s (%s)" % (v.title(), units))
                ax.flat[i].set_xlabel("Height (Mm)")
                if i == 0:
                    ax.flat[i].set_title(os.path.split(self.filename)[1])
                if i == 1:
                    ax.flat[i].set_title("snapshot=%i, x=%i, y=%i" % (0, 0, 0))
            fig.tight_layout()
            return ax

        ax = _atmos_plot()

        @interact(snapshot=tslider, x=x_slider, y=y_slider, y_log=True)
        def _atmos_update(snapshot=0, x=0, y=0, y_log=True):
            for i, v in enumerate(self.variables):
                var = self.atmos.variables[v]
                if v[:8].lower() == 'velocity':  # to km/s
                    ydata = var[snapshot, x, y] / 1.e3
                elif v.lower() == "hydrogen_populations":
                    ydata = var[snapshot, :, x, y].sum(axis=0)
                else:
                    ydata = var[snapshot, x, y]
                ax.flat[i].lines[0].set_ydata(ydata)
                if len(self.atmos.z.shape) == 2:
                    zdata = self.atmos.z[snapshot] / 1e6
                elif len(self.atmos.z.shape) == 4:
                    zdata = self.atmos.z[snapshot, x, y] / 1e6
                else:
                    raise ValueError("Invalid shape of z array")
                ax.flat[i].lines[0].set_xdata(zdata)
                ax.flat[i].relim()
                ax.flat[i].autoscale_view(True, True, True)
                if i == 1:
                    tmp = "snapshot=%i, x=%i, y=%i" % (snapshot, x, y)
                    ax.flat[i].set_title(tmp)
                if v[:2].lower() not in ['ve', 'b_']:  # no log in v and B
                    if y_log:
                        ax.flat[i].set_yscale("log")
                    else:
                        ax.flat[i].set_yscale("linear")
