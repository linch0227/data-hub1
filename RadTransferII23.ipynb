{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import ipywidgets as widgets\n",
    "from matplotlib.widgets import Slider\n",
    "from matplotlib import cm\n",
    "from ipywidgets import interact, interactive, fixed, interact_manual\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute(I0, S, tauc, taul):\n",
    "    \n",
    "    # damping parameter\n",
    "    damping = 0.5\n",
    "    \n",
    "    # frequency range of nu-nu_0\n",
    "    nu = np.linspace(-6, 6, 1000)\n",
    "    \n",
    "    # Lorentz distribution\n",
    "    lorentz =  1/np.pi * damping / ( np.power(nu,2) + np.power(damping, 2)) \n",
    "     \n",
    "    # normalize and weight total tau with the Lorentz distribution\n",
    "    tau = (lorentz * taul) + tauc \n",
    "    \n",
    "    # intensity formula for a homogeneous medium\n",
    "    I = I0 * np.exp(-tau) + S*(1 - np.exp(-tau))\n",
    "    \n",
    "    return nu, I\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_plot(I0=10.,S=50.,tauc=1.,taul=1., printplot=False):\n",
    "    # make space for the sliders\n",
    "    fig = plt.figure(111)\n",
    "    ax = fig.add_subplot(111)\n",
    "    #plt.subplots_adjust(bottom=0.35)\n",
    "\n",
    "    # compute and plot initial line\n",
    "    nu, I = compute(I0, S, tauc, taul)\n",
    "    #\n",
    "#   ax.set_xscale(\"log\", nonposx='clip')\n",
    "    ax.plot(nu,I, 'k', lw=2, alpha=0.4)\n",
    "    \n",
    "    # label the plot\n",
    "    plt.title(\"Radiative Transport Through a Slab\")\n",
    "    plt.xlabel(r\"$\\nu/\\nu_0$\", fontsize=16)\n",
    "    plt.ylabel(r\"$I_\\nu(D)$\", fontsize=16)\n",
    "    plt.ylim(0, 100)\n",
    " #   plt.ylim(0.9*I0,1.05*S)\n",
    "    plt.xlim(-6,6)\n",
    "    ax.text(-5.5, 95, r'$I(0) = $'+str(I0), fontsize=9)\n",
    "    ax.text(-5.5, 90, r'$S = $'+str(S), fontsize=9)\n",
    "    ax.text(-5.5, 85, r'$\\tau_{\\rm c} = $'+str(tauc), fontsize=9)\n",
    "    ax.text(-5.5, 80, r'$\\tau_{\\rm l} = $'+str(taul), fontsize=9)\n",
    "    ax.plot(nu, S*np.power(nu,0),'--', lw=2, alpha=0.4)\n",
    "    ax.plot(nu, I0*np.power(nu,0),'--', lw=2, alpha=0.4)\n",
    "    #\n",
    "\n",
    "    if printplot:\n",
    "       fig.savefig(\"figure_\"+str(int(round(I0)))+\"_\"+str(int(round(S)))+\"_\"+str(int(round(tauc)))+\"_\"+str(int(round(taul)))+\".pdf\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Radiate():\n",
    "    \n",
    "    interact(make_plot,I0=(0.,100.,1),S=(0.,100.,1),tauc=(0.,10.,.1),taul=(0.,20.,.1), printplot=False)\n",
    "\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5df8bb35050b490d96c83b13183437aa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "aW50ZXJhY3RpdmUoY2hpbGRyZW49KEZsb2F0U2xpZGVyKHZhbHVlPTEwLjAsIGRlc2NyaXB0aW9uPXUnSTAnLCBzdGVwPTEuMCksIEZsb2F0U2xpZGVyKHZhbHVlPTUwLjAsIGRlc2NyaXB0aW/igKY=\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# main function (called if not imported as module)\n",
    "if __name__==\"__main__\":\n",
    "    Radiate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
