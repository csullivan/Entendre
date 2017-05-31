import sys
import matplotlib.pyplot as plt
import numpy as np

def setfont(font='helvetica',unicode=True):
    r"""
    Set Matplotlibs rcParams to use LaTeX for font rendering.
    Revert all changes by calling rcdefault() from matplotlib.

    Parameters:
    -----------
    font: string
        "Helvetica"
        "Times"
        "Computer Modern"

    usetex: Boolean
        Use unicode. Default: False.
    """

    # Use TeX for all figure text!
    plt.rc('text', usetex=True)

    font = font.lower().replace(" ","")
    if font == 'times':
        # Times
        font = {'family':'serif', 'serif':['Times']}
        preamble  = r"""
                       \usepackage{color}
                       \usepackage{mathptmx}
                    """
    elif font == 'helvetica':
        # Helvetica
        # set serif, too. Otherwise setting to times and then
        # Helvetica causes an error.
        font = {'family':'sans-serif','sans-serif':['Helvetica'],
                'serif':['cm10']}
        preamble  = r"""
                       \usepackage{color}
                       \usepackage[tx]{sfmath}
                       \usepackage{helvet}
                    """
    else:
        # Computer modern serif
        font = {'family':'serif', 'serif':['cm10']}
        preamble  = r"""
                       \usepackage{color}
                    """

    if unicode:
        # Unicode for Tex
        #preamble =  r"""\usepackage[utf8]{inputenc}""" + preamble
        # inputenc should be set automatically
        plt.rcParams['text.latex.unicode']=True

    #print font, preamble
    plt.rc('font',**font)
    plt.rcParams['text.latex.preamble'] = preamble


def setticks(ax,xlog=False,ylog=False,xmajor=5,xminor=1,ymajor=2,yminor=0.5):

    if not xlog:
        xmajorLocator   = plt.MultipleLocator(xmajor)
        xmajorFormatter = plt.FormatStrFormatter('%d')
        xminorLocator   = plt.MultipleLocator(xminor)
        ax.xaxis.set_major_locator(xmajorLocator)
        #ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.xaxis.set_minor_locator(xminorLocator)

    if not ylog:
        ymajorLocator   = plt.MultipleLocator(ymajor)
        ymajorFormatter = plt.FormatStrFormatter('%d')
        yminorLocator   = plt.MultipleLocator(yminor)
        ax.yaxis.set_major_locator(ymajorLocator)
        #ax.yaxis.set_major_formatter(ymajorFormatter)
        ax.yaxis.set_minor_locator(yminorLocator)

    ax.get_yaxis().set_tick_params(which='both', direction='out')
    ax.get_xaxis().set_tick_params(which='both', direction='out')
    for tick in ax.xaxis.get_ticklines():
        tick.set_markersize(3.5)
    for tick in ax.yaxis.get_ticklines():
        tick.set_markersize(3.5)
    for tick in ax.xaxis.get_ticklines(minor=True):
        tick.set_markersize(2.5)
    for tick in ax.yaxis.get_ticklines(minor=True):
        tick.set_markersize(2.5)




if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print "Error: Enter filename as program argument."
        exit()

    num_nets = []
    speedup = []
    subset = []
    unique_num_nets = []
    avgs = []
    stddevs = []
    current_num_net = 0
    for line in open(filename):
        if 'Expanded genome' in line:
            number_of_nets = int(line.split()[-2][:-1])
            if current_num_net != number_of_nets:
                if current_num_net != 0:
                    avgs.append(np.average(subset))
                    stddevs.append(np.std(subset))
                    unique_num_nets.append(current_num_net)
                subset = []
                current_num_net = number_of_nets
            num_nets.append(number_of_nets)
        if 'GPU speed up' in line:
            speedup_factor = float(line.split()[-2])
            subset.append(speedup_factor)
            speedup.append(speedup_factor)


    avgs.append(np.average(subset))
    stddevs.append(np.std(subset))
    unique_num_nets.append(current_num_net)

    setfont()
    fig,axes = plt.subplots(1)
    #axes.scatter(num_nets,speedup,color='red')
    axes.scatter(unique_num_nets,avgs,color='black',zorder=2,s=2)
    axes.errorbar(unique_num_nets,avgs,yerr=stddevs,ls='none',color='black',elinewidth=1,capsize=0,zorder=3)

    axes.set_xscale('log',basex=2)
    setticks(axes,xlog=True,ylog=True)
    plt.savefig("/user/sullivan/public_html/speedup_full.pdf")
