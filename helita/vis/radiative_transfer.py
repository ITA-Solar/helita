"""
Set of functions and widgets for radiative transfer visualisations
"""
import warnings
import numpy as np
from pkg_resources import resource_filename
from scipy import interpolate as interp
import bqplot.pyplot as plt
from bqplot import LogScale
from ipywidgets import (interactive, Layout, HBox, VBox, Box, GridBox,
                        IntSlider, FloatSlider, Dropdown, HTMLMath)
from ..utils.utilsmath import voigt


def transp():
    """
    Instantiates the Transp() class, and shows the widget.
    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return Transp().widget


class Transp():
    """
    Class for a widget illustrating line formation given a source function,
    Voigt profile and opacity.

    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    """
    DATAFILE = resource_filename('helita', 'data/VAL3C_source_functions.npz')
    data = np.load(DATAFILE)
    # variable names inside data structure
    SFUNCTIONS = {"VAL3C Mg": "s_nu_mg", "VAL3C Ca": "s_nu_ca",
                  "VAL3C LTE": "s_nu_lte"}
    TAUS =  {"VAL3C Mg": "t_500_mg", "VAL3C Ca": "t_500_ca",
             "VAL3C LTE": "t_500_lte"}
    # initial parameters
    mu = 1.0
    npts = 101
    xmax = 50
    a = -2.5
    opa_cont = 0.
    opa_line = 6.44
    source = "VAL3C Mg"

    def __init__(self):
        self._compute_profile()
        self._make_plot()
        self._make_widget()

    def _compute_profile(self):
        """
        Calculates the line profile given a a damping parameter,
        source function, opacities, and mu.
        """
        self.tau500 = self.data[self.TAUS[self.source]]
        self.source_function = self.data[self.SFUNCTIONS[self.source]]
        tau500 = self.tau500
        source_function = self.source_function
        self.freq = np.linspace(-float(self.xmax), self.xmax, self.npts)
        a = 10. ** self.a
        self.h = voigt(a, self.freq)
        self.xq = self.h * 10. ** self.opa_line + 10. ** self.opa_cont
        xq = self.xq
        self.tau500_cont = self.mu / 10 ** self.opa_cont
        self.tau500_line = self.mu / self.xq.max()
        f = interp.interp1d(tau500, source_function, bounds_error=False)
        self.source_function_cont = f(self.tau500_cont)[()]
        self.source_function_line = f(self.tau500_line)[()]
        xq = xq[:, np.newaxis]
        tmp = source_function * np.exp(-xq * tau500 / self.mu) * xq * tau500
        self.prof = np.log(10) / self.mu * np.trapz(tmp.T, np.log(tau500),
                                                    axis=0)

    def _make_plot(self):
        plt.close(1)
        fig_margin = {'top': 25, 'bottom': 35, 'left': 35, 'right':25}
        fig_layout = {'height': '100%', 'width': '100%' }
        layout_args = {'fig_margin': fig_margin, 'layout': fig_layout,
                       'max_aspect_ratio': 1.618}
        self.voigt_fig = plt.figure(1, title='Voigt profile', **layout_args)
        self.voigt_plot = plt.plot(self.freq, self.h, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")

        plt.close(2)
        self.abs_fig = plt.figure(2, title='(αᶜ + αˡ) / α₅₀₀', **layout_args)
        self.abs_plot = plt.plot(self.freq, self.xq, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")

        plt.close(3)
        self.int_fig = plt.figure(3, title='Intensity', **layout_args)
        self.int_plot = plt.plot(self.freq, self.prof, scales={'y': LogScale()})
        plt.xlabel("Δν / ΔνD")

        plt.close(4)
        self.source_fig = plt.figure(4, title='Source Function', **layout_args)
        self.source_plot = plt.plot(np.log10(self.tau500), self.source_function,
                                    scales={'y': LogScale()})
        plt.xlabel("lg(τ₅₀₀)")
        self.tau_labels = plt.label(['τᶜ = 1', 'τˡ = 1'], colors=['black'],
                                    x=np.array([np.log10(self.tau500_cont),
                                                np.log10(self.tau500_line)]),
                                    y=np.array([self.source_function_cont,
                                                self.source_function_line]),
                                    y_offset=-25, align='middle')
        self.tau_line_plot = plt.plot(np.array([np.log10(self.tau500_line),
                                                np.log10(self.tau500_line)]),
                                      np.array([self.source_function_line / 1.5,
                                                self.source_function_line * 1.5]),
                                      colors=['black'])
        self.tau_cont_plot = plt.plot(np.array([np.log10(self.tau500_cont),
                                                np.log10(self.tau500_cont)]),
                                      np.array([self.source_function_cont / 1.5,
                                                self.source_function_cont * 1.5]),
                                      colors=['black'])

    def _update_plot(self, a, opa_cont, opa_line, mu, xmax, source):
        self.a = a
        self.opa_cont = opa_cont
        self.opa_line = opa_line
        self.mu = mu
        self.xmax = xmax
        self.source = source
        self._compute_profile()
        self.voigt_plot.x = self.freq
        self.voigt_plot.y = self.h
        self.abs_plot.x = self.freq
        self.abs_plot.y = self.xq
        self.int_plot.x = self.freq
        self.int_plot.y = self.prof
        self.source_plot.x = np.log10(self.tau500)
        self.source_plot.y = self.source_function
        self.tau_labels.x = np.array([np.log10(self.tau500_cont),
                                      np.log10(self.tau500_line)])
        self.tau_labels.y = np.array([self.source_function_cont,
                                      self.source_function_line])
        self.tau_line_plot.x = [np.log10(self.tau500_line),
                                np.log10(self.tau500_line)]
        self.tau_line_plot.y = [self.source_function_line / 1.5,
                                self.source_function_line * 1.5]
        self.tau_cont_plot.x = [np.log10(self.tau500_cont),
                                np.log10(self.tau500_cont)]
        self.tau_cont_plot.y = [self.source_function_cont / 1.5,
                                self.source_function_cont * 1.5]


    def _make_widget(self):
        fig = GridBox(children=[self.voigt_fig, self.abs_fig,
                                self.int_fig, self.source_fig],
                      layout=Layout(width='100%',
                                    min_height='600px',
                                    height='100%',
                                    grid_template_rows='49% 49%',
                                    grid_template_columns='49% 49%',
                                    grid_gap='0px 0px'))

        a_slider = FloatSlider(min=-5, max=0., step=0.01, value=self.a,
                               description='lg(a)')
        opa_cont_slider = FloatSlider(min=0., max=6., step=0.01,
                 value=self.opa_cont, description=r"$\kappa_c / \kappa_{500}$")
        opa_line_slider = FloatSlider(min=0., max=7., step=0.01,
                 value=self.opa_line, description=r"$\kappa_l / \kappa_{500}$")
        mu_slider = FloatSlider(min=0.01, max=1., step=0.01,
                                value=self.mu, description=r'$\mu$')
        xmax_slider = IntSlider(min=1, max=100, step=1, value=self.xmax,
                                description='xmax')
        source_slider = Dropdown(options=self.SFUNCTIONS.keys(), value=self.source,
                                 description='Source Function',
                                 style={'description_width': 'initial'})
        w = interactive(self._update_plot, a=a_slider, opa_cont=opa_cont_slider,
                        opa_line=opa_line_slider, mu=mu_slider,
                        xmax=xmax_slider, source=source_slider)
        controls = GridBox(children=[w.children[5], w.children[0],
                                     w.children[2], w.children[4],
                                     w.children[3], w.children[1]],
                           layout=Layout(min_height='80px',
                                         min_width='600px',
                                         grid_template_rows='49% 49%',
                                         grid_template_columns='31% 31% 31%',
                                         grid_gap='10px'))
        self.widget = GridBox(children=[controls, fig],
                              layout=Layout(grid_template_rows='8% 90%',
                                            width='100%',
                                            min_height='650px',
                                            height='100%',
                                            grid_gap='10px'))


def slab():
    """
    Displays a widget illustrating line formation in a homogenous slab.

    Runs only in Jupyter notebook or JupyterLab. Requires bqplot.
    """
    # Don't display some ipywidget warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    def _compute_slab(i0, source, tau_cont, tau_line):
        """
        Calculates slab line profile.
        """
        NPT = 101
        MAX_DX = 5.
        x = np.arange(NPT) - (NPT - 1.) / 2
        x *= MAX_DX / x.max()
        tau = tau_cont + tau_line * np.exp(-x * x)
        extinc = np.exp(-tau)
        intensity = float(i0) * extinc + float(source) * (1. - extinc)
        return (x, intensity)

    I0 = 15
    S = 65
    x, y = _compute_slab(I0, S, 0.5, 0.9)
    base = np.zeros_like(x)
    fig = plt.figure(title='Slab line formation')
    int_plot = plt.plot(x, y, 'b-')
    source_line = plt.plot(x, base + S, 'k--')
    i0_line = plt.plot(x, base + I0, 'k:')
    labels = plt.label(['I₀', 'I', 'S'],
                       x=np.array([int_plot.x[0] + 0.2, int_plot.x[-1] - 0.2,
                                   int_plot.x[0] + 0.2]),
                       y=np.array([i0_line.y[0], int_plot.y[0],
                                   source_line.y[0]]) + 2,
                       colors=['black'])
    plt.ylim(0, 100)
    i0_slider = IntSlider(min=0, max=100, value=I0, description=r'$I_0$')
    s_slider = IntSlider(min=0, max=100, value=S, description=r'$S$')
    tau_c_slider = FloatSlider(min=0, max=1., step=0.01, value=0.5,
                               description=r'$\tau_{\mathrm{cont}}$')
    tau_l_slider = FloatSlider(min=0, max=10., step=0.01, value=0.9,
                               description=r'$\tau_{\mathrm{line}}$')

    def plot_update(i0=I0, source=S, tau_cont=0.5, tau_line=0.9):
        _, y = _compute_slab(i0, source, tau_cont, tau_line)
        int_plot.y = y
        source_line.y = base + source
        i0_line.y = base + i0
        labels.y = np.array([i0, y[0], source]) + 2

    widg = interactive(plot_update, i0=i0_slider, source=s_slider,
                       tau_cont=tau_c_slider, tau_line=tau_l_slider)
    help_w = HTMLMath("<p><b>Purpose: </b>"
      "This widget-based procedure is used for "
      "studying spectral line formation in a "
      "homogeneous slab.</p>"
      "<p><b>Inputs:</b></p>"
      "<ul>"
      r"   <li>$I_0$: The incident intensity.</li>"
      r"   <li>$S$: The source function.</li>"
      r"   <li>$\tau_{\mathrm{cont}}$ : The continuum optical depth.</li>"
      r"   <li>$\tau_{\mathrm{line}}$ : The integrated optical depth in the spectral line.</li>"
      "</ul>")
    return HBox([VBox([widg, help_w],
                      layout=Layout(width='33%', top='50px', left='5px')),
                 Box([fig], layout=Layout(width='66%'))],
                layout=Layout(border='50px'))
