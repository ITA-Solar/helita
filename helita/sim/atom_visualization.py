
import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
from .atom_tools import norm_ne_goftab, add_goftab
from scipy.io import readsav

def fig_norm_ne_goftab_all(response_func=None,norm=True):

    files = glob.glob("./*.opy")

    nfiles = np.size(files)
    for ifile in range(0, nfiles):
        fileh = open(files[ifile], 'rb')
        print(files[ifile])
        table = pickle.load(fileh)
        goftab = table.Gofnt['gofnt']
        if norm:
            tabnorm = norm_ne_goftab(goftab)
            vmin=np.min(tabnorm)
        else:
            tabnorm = np.log10(goftab)
            vmin=-50
        fig = plt.figure()
        extent = np.log10((np.min(table.Gofnt['press']), np.max(
            table.Gofnt['press']), np.min(
            table.Gofnt['temperature']), np.max(table.Gofnt['temperature'])))
        im = plt.imshow(tabnorm, extent=extent, aspect='auto',vmin=vmin)
        plt.colorbar(im)
        plt.title(files[ifile][2:-4])
        plt.ylabel("Temperature")
        plt.xlabel("Pressure")
        if norm:
            fig.savefig('pressure_dependence_%s.png' %
                    files[ifile][2:-4], bbox_inches="tight", format="png",
                    dpi=650)
        else:
            fig.savefig('pressure_dependence_%sint.png' %
                    files[ifile][2:-4], bbox_inches="tight", format="png",
                    dpi=650)

        fileh.close()
        plt.close("all")

def print_all_max(response_func=None):

    files = glob.glob("./*.opy")

    nfiles = np.size(files)
    for ifile in range(0, nfiles):
        fileh = open(files[ifile], 'rb')
        table = pickle.load(fileh)
        print(files[ifile], np.max(table.Gofnt['gofnt']))
        fileh.close()

def print_comp_lines(response_func=None):

    files = glob.glob("./*.opy")

    nfiles = np.size(files)
    for ifile in range(0, nfiles):
        fileh = open(files[ifile], 'rb')
        table = pickle.load(fileh)
        fileb = open(files[ifile], 'rb')
        high = pickle.load(fileb)
        high.gofnt(wvlRange=[table.Gofnt['wvl'] - 0.1,
                             table.Gofnt['wvl'] + 0.1], top=1, plot=False)
        print(files[ifile], table.Gofnt['wvl'], high.Gofnt['wvl'])
        fileh.close()
        fileb.close()

def fig_norm_ne_goftab_comb(namelist="./*_*_10?.*.opy",response_func=None,nametitle='108',norm=True):

    files = glob.glob(namelist)
    goftab = add_goftab(files,response_func=response_func)
    nfiles = np.size(files)
    if norm:
        tabnorm = norm_ne_goftab(goftab)
        minv = np.min(tabnorm)
    else:
        tabnorm = np.log10(goftab)
        minv = -40
    fileh = open(files[0], 'rb')
    print(files[0])
    table = pickle.load(fileh)

    fig = plt.figure()
    extent = np.log10((np.min(table.Gofnt['press']), np.max(
        table.Gofnt['press']), np.min(
        table.Gofnt['temperature']), np.max(table.Gofnt['temperature'])))
    im = plt.imshow(tabnorm, extent=extent, aspect='auto',vmin=minv)
    plt.colorbar(im)
    plt.title(nametitle)
    plt.ylabel("Temperature")
    plt.xlabel("Pressure")
    if response_func is None:
        if norm:
            fig.savefig('All_pressure_dependence_%s.png' %
                nametitle, bbox_inches="tight", format="png", dpi=650)
        else:
            fig.savefig('All_pressure_dependence_%s_int.png' %
                nametitle, bbox_inches="tight", format="png", dpi=650)
    else:
        if norm:
            fig.savefig('All_pressure_dependence_%sresp_sji.png' %
                nametitle, bbox_inches="tight", format="png", dpi=650)
        else:
            fig.savefig('All_pressure_dependence_%sresp_sji_int.png' %
                nametitle, bbox_inches="tight", format="png", dpi=650)
