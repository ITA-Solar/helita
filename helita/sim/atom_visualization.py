
import pickle
import glob
import matplotlib.pyplot as plt


def fig_norm_ne_goftab_all():

    files = glob.glob("./*.opy")

    nfiles = np.size(files)
    for ifile in range(0, nfiles):
        fileh = open(files[ifile], 'rb')
        print(files[ifile])
        table = pickle.load(fileh)
        goftab = table.Gofnt['gofnt']
        tabnorm = norm_ne_goftab(goftab)
        fig = plt.figure()
        extent = np.log10((np.min(table.Gofnt['press']), np.max(
            table.Gofnt['press']), np.min(
            table.Gofnt['temperature']), np.max(table.Gofnt['temperature'])))
        im = plt.imshow(tabnorm, extent=extent, aspect='auto')
        plt.colorbar(im)
        plt.title(files[ifile][2:-4])
        plt.ylabel("Temperature")
        plt.xlabel("Pressure")
        fig.savefig('pressure_dependence_%s.png' %
                    files[ifile][2:-4], bbox_inches="tight", format="png",
                    dpi=650)


def print_all_max():

    files = glob.glob("./*.opy")

    nfiles = np.size(files)
    for ifile in range(0, nfiles):
        fileh = open(files[ifile], 'rb')
        table = pickle.load(fileh)
        print(files[ifile], np.max(table.Gofnt['gofnt']))


def print_comp_lines():

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


def fig_norm_ne_goftab_comb(namelist="./*_10?.*.opy"):

    files = glob.glob(namelist)
    goftab = add_goftab(files)
    nfiles = np.size(files)
    tabnorm = norm_ne_goftab(goftab)

    fileh = open(files[0], 'rb')
    print(files[0])
    table = pickle.load(fileh)

    fig = plt.figure()
    extent = np.log10((np.min(table.Gofnt['press']), np.max(
        table.Gofnt['press']), np.min(
        table.Gofnt['temperature']), np.max(table.Gofnt['temperature'])))
    im = plt.imshow(tabnorm, extent=extent, aspect='auto')
    plt.colorbar(im)
    plt.title(namelist[4:-7])
    plt.ylabel("Temperature")
    plt.xlabel("Pressure")
    fig.savefig('All_pressure_dependence_%s.png' %
                namelist[3:-5], bbox_inches="tight", format="png", dpi=650)
