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
from ..utils.utilsmath import planck, int2bt, voigt


class Populations:
    def __init__(self, rh_object):
        self.rhobj = rh_object
        self.atoms = [a for a in dir(self.rhobj) if a[:5] == 'atom_']
        self.display()

    def display(self):
        """
        Displays a graphical widget to explore the level populations.
        Works in jupyter only.
        """
        ATOMS = {a.split('_')[1].title(): a for a in self.atoms}
        QUANTS = ['Populations', 'LTE Populations', 'Departure coefficients']
        NLEVEL = getattr(self.rhobj, self.atoms[0]).nlevel
        NX, NY, NZ = self.rhobj.atmos.temperature.shape
        if NX == 1:
            xslider = fixed(0)
        else:
            xslider = (0, NX - 1)
        if NY == 1:
            yslider = fixed(0)
        else:
            yslider = (0, NY - 1)

        def _pop_plot(atom):
            """Starts population plot"""
            pop = getattr(self.rhobj, atom).populations
            height = self.rhobj.atmos.height_scale[0, 0] / 1e6  # in Mm
            fig, ax = plt.subplots()
            pop_plot, = ax.plot(height, pop[0, 0, 0])
            ax.set_xlabel("Height (Mm)")
            ax.set_ylabel("Populations")
            ax.set_title("Level 1")
            return ax, pop_plot

        ax, p_plot = _pop_plot(self.atoms[0])

        @interact(atom=ATOMS, quantity=QUANTS, y_log=False,
                  x=xslider, y=xslider)
        def _pop_update(atom, quantity, y_log=False, x=0, y=0):
            NLEVEL = getattr(self.rhobj, atom).nlevel

            # Atomic level singled out because NLEVEL depends on the atom
            @interact(level=(1, NLEVEL))
            def _pop_update_level(level=1):
                n = getattr(self.rhobj, atom).populations[level - 1, x, y]
                nstar = getattr(self.rhobj, atom).populations_LTE[level - 1, x, y]
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
    def __init__(self, rh_object):
        self.rhobj = rh_object
        self.display()

    def display(self):
        """
        Displays a graphical widget to explore the source function.
        Works in jupyter only.
        """
        NX, NY, NZ, NWAVE = self.rhobj.ray.source_function.shape
        if NX == 1:
            xslider = fixed(0)
        else:
            xslider = (0, NX - 1)
        if NY == 1:
            yslider = fixed(0)
        else:
            yslider = (0, NY - 1)
        TAU_LEVELS = [0.3, 1., 3.]
        ARROW = dict(facecolor='black', width=1., headwidth=5, headlength=6)
        SCALES = ['Height', 'Optical depth']

        def __get_tau_levels(x, y, wave):
            """
            Calculates height where tau=0.3, 1., 3 for a given
            wavelength index.
            Returns height in Mm and closest indices of height array.
            """
            h = self.rhobj.atmos.height_scale[x, y].dropna('height')
            tau = cumtrapz(self.rhobj.ray.chi[x, y, :, wave].dropna('height'),
                           x=-h)
            tau = interp1d(tau, h[1:])(TAU_LEVELS)
            idx = np.around(interp1d(h, np.arange(h.shape[0]))(tau)).astype('i')
            return (tau / 1e6, idx)  # in Mm

        def _sf_plot():
            """Starts source function plot"""
            obj = self.rhobj
            sf = obj.ray.source_function[0, 0, :, 0].dropna('height')
            height = obj.atmos.height_scale[0, 0].dropna('height') / 1e6  # in Mm
            bplanck = planck(obj.ray.wavelength_selected[0],
                             obj.atmos.temperature[0, 0].dropna('height'),
                             units='Hz')
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
            for i, level in enumerate(TAU_LEVELS):
                xval = tau_v[i]
                yval = sf[h_idx[i]]
                ax.annotate(r'$\tau$=%s' % level,
                            xy=(xval, yval),
                            xytext=(xval, yval / (0.2 - 0.03 * i)),
                            arrowprops=ARROW, ha='center', va='top')
            return ax

        ax = _sf_plot()

        @interact(wavelength=(0, NWAVE - 1, 1), y_log=True,
                  x=xslider, y=xslider)
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
            for i in range(len(TAU_LEVELS)):
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
        NT, NX, NY, NZ = self.atmos.temperature.shape
        if NT == 1:
            tslider = fixed(0)
        else:
            tslider = (0, NT - 1)
        if NX == 1:
            xslider = fixed(0)
        else:
            xslider = (0, NX - 1)
        if NY == 1:
            yslider = fixed(0)
        else:
            yslider = (0, NY - 1)

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

        @interact(snapshot=tslider, x=xslider, y=xslider, y_log=True)
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

def slab():
    """
    Displays a graphical widget to demonstrate radiative transfer
    in a homogeneous slab. Based on IDL routine xslab.pro.
    """
    def _get_slab_intensity(i0, sf, tau_c, tau_l):
        NPTS = 101
        MAX_DX = 5.
        x = np.arange(NPTS) - (NPTS - 1.) / 2
        x *= MAX_DX / x.max()
        tau = tau_c + tau_l * np.exp(-x * x)
        extinc = np.exp(-tau)
        return (x, float(i0) * extinc + float(sf) * (1. - extinc))

    def _slab_plot():
        x, intensity = _get_slab_intensity(10, 65, 0.5, 0.9)
        fig, ax = plt.subplots()
        ax.axhline(y=65, color='k', ls='--', lw=1, label='S')
        ax.plot(x, intensity, 'b-', lw=1, label='I')
        ax.axhline(y=10, color='k', ls=':', lw=1, label='I$_0$')
        ax.set_ylim(-2, 102)
        ax.legend(loc='upper right')
        ax.set_title("Spectral line formation in homogeneous slab")
        return ax

    ax = _slab_plot()

    style = {'description_width': 'initial'}
    i0s = IntSlider(value=10, min=0, max=100, step=1, description='I$_0$')
    sfs = IntSlider(value=65, min=0, max=100, step=1,
                   description='Source Function', style=style)
    tau_cs = FloatSlider(value=0.5, min=0., max=1., step=0.01,
                        description=r'$\tau_{\mathrm{cont}}$')
    tau_ls = FloatSlider(value=0.9, min=0., max=10., step=0.1,
                        description=r'$\tau_{\mathrm{line}}$')
    @interact(i0=i0s, sf=sfs, tau_c=tau_cs, tau_l=tau_ls)
    def _slab_update(i0=10, sf=65, tau_c=0.5, tau_l=0.9):
        x, intensity = _get_slab_intensity(i0, sf, tau_c, tau_l)
        ax.lines[1].set_xdata(x)
        ax.lines[1].set_ydata(intensity)
        ax.lines[2].set_ydata([i0, i0])
        ax.lines[0].set_ydata([sf, sf])


def transp():
    """
    Displays a graphical widget to demonstrate spectral line formation in
    a 1D model atmosphere. Based on IDL routine xtransp.pro.
    """
    def _get_profile(tau500, sf, a, mu, opa_cont, opa_line, xmax):
        NPTS = 101
        v = np.linspace(-float(xmax), xmax, NPTS)
        a = 10. ** a
        h = voigt(a, v)
        xq = h * 10. ** opa_line + 10. ** opa_cont
        tau500_cont = mu / 10. ** opa_cont
        tau500_line = mu / xq.max()
        f = interp1d(tau500, sf, bounds_error=False)
        sf_cont = f(tau500_cont)[()]
        sf_line = f(tau500_line)[()]
        xq = xq[:, np.newaxis]
        tmp = sf * np.exp(-xq * tau500 / mu) * xq * tau500
        prof = np.log(10) / mu * np.trapz(tmp.T, np.log(tau500), axis=0)
        return (v, h, xq, prof, tau500_cont, tau500_line, sf_cont, sf_line)

    def _transp_plot():
        tau500 = data['t_500_mg']
        source_function = data['s_nu_mg']
        tmp = _get_profile(tau500, source_function, -2.5, 1., 0., 6.44, 50)
        v, h, xq, prof, tau500_cont, tau500_line, sf_cont, sf_line = tmp

        fig = plt.figure(figsize=(7,5))
        ax = []
        # Voigt profile plot
        ax.append(fig.add_subplot(2, 2, 1))
        ax[0].plot(v, h, lw=1)
        ax[0].set_yscale("log")
        ax[0].set_title("Voigt profile")
        ax[0].set_xlabel(r"$\Delta\nu/\Delta\nu_0$")
        # Opacity plot
        ax.append(fig.add_subplot(2, 2, 2, sharex = ax[0]))
        ax[1].plot(v, xq, lw=1)
        ax[1].set_yscale("log")
        ax[1].set_title(r"($\alpha_c$ + $\alpha_l$) / $\alpha_{500}$")
        ax[1].set_xlabel(r"$\Delta\nu/\Delta\nu_0$")
        # Intensity plot
        ax.append(fig.add_subplot(2, 2, 3, sharex = ax[0]))
        ax[2].plot(v, prof, lw=1)
        ax[2].set_ylabel(r"I$_\nu$")
        ax[2].set_xlabel(r"$\Delta\nu/\Delta\nu_0$")
        ax[2].ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        # Source function plot
        ax.append(fig.add_subplot(2, 2, 4))
        ax[3].plot(np.log10(tau500), source_function, lw=1)
        ax[3].set_yscale("log")
        ax[3].set_ylabel(r"S$_\nu$")
        ax[3].set_xlabel(r"log($\tau_{500}$)")
        ARROW = dict(facecolor='black', width=1., headwidth=5, headlength=6)
        for tau, sf, label in zip([tau500_cont, tau500_line],
                                  [sf_cont, sf_line], ['_c', '_l']):
            ax[3].annotate(r'$\tau%s$=1' % label,
                        xy=(np.log10(tau), sf),
                        xytext=(np.log10(tau), sf / 0.19),
                        arrowprops=ARROW, ha='center', va='top')
        plt.tight_layout()
        return ax

    DATAFILE = resource_filename('helita', 'data/VAL3C_source_functions.npz')
    data = np.load(DATAFILE)
    ax = _transp_plot()

    SOURCE_FUNCS = ['VAL3C Ca', 'VAL3C Mg', 'VAL3C LTE']
    style = {'description_width': 'initial'}
    sfs = Dropdown(value='VAL3C Mg',options=SOURCE_FUNCS,
                   description='Source Function', style=style)
    opa_cs = FloatSlider(value=0.5, min=0., max=6., step=0.05,
                        description=r'$\chi_{\mathrm{cont}}$')
    opa_ls = FloatSlider(value=6.44, min=0., max=7., step=0.05,
                        description=r'$\chi_{\mathrm{line}}$')
    mus = FloatSlider(value=1.00, min=0.001, max=1.001, step=0.05,
                        description=r'$\mu$')
    @interact(source=sfs, a=(-5., 0., 0.05), mu=mus,opa_cont=opa_cs,
              opa_line=opa_ls, xmax=(1, 100, 1), continuous_update=False)
    def _transp_update(source='VAL3C Mg', a=-2.5, mu=1., opa_cont=0.,
                       opa_line=6.44, xmax=50):
        key = source.split()[1].lower()
        tau500 = data['t_500_' + key]
        source_function = data['s_nu_' + key]
        tmp = _get_profile(tau500, source_function, a, mu,
                           opa_cont, opa_line, xmax)
        v, h, xq, prof, tau500_cont, tau500_line, sf_cont, sf_line = tmp
        ax[0].lines[0].set_xdata(v)
        ax[0].lines[0].set_ydata(h)
        ax[1].lines[0].set_xdata(v)
        ax[1].lines[0].set_ydata(xq)
        ax[2].lines[0].set_xdata(v)
        ax[2].lines[0].set_ydata(prof)
        ax[3].lines[0].set_xdata(np.log10(tau500))
        ax[3].lines[0].set_ydata(source_function)
        for i, tau, sf, label in zip([0, 1], [tau500_cont, tau500_line],
                                     [sf_cont, sf_line], ['_c', '_l']):
            ax[3].texts[i].xy = (np.log10(tau), sf)
            ax[3].texts[i].set_position((np.log10(tau), sf / 0.19))
        for a in ax:
            a.relim()
            a.autoscale_view(True, True, True)
