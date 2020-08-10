from collections import OrderedDict
from functools import reduce
from glob import glob
from operator import add

import matplotlib.pyplot as plt
import numpy as np
from PyCAR.PyCIT.FT import PowDens
from aer_construction import AerModel

from citvappru.SourceCAREM2_1_1 import Geometry

def main():
    alf = AerModel()
    elements = [hexag.type[0] for hexag in
                list(filter(lambda column: 26 in map(lambda hexag: hexag.type[1], column), alf.grid))[0] if
                hexag.type[1] not in [5, 1999]]
    pows = PowDens('aer.cdb.powerdensities')
    plt.plot(pows[0, min(elements) - 1:max(elements) - 1, :], '-o')

    Pows = OrderedDict({ file.strip('.pow') : np.genfromtxt(file) for file in glob('INFO\\*.pow')})

    Nods = OrderedDict({file.strip('.nod'): np.genfromtxt(file, comments='*') for file in glob('INFO\\*.nod')})
    Nods = OrderedDict({ key:value.reshape(value.shape[0]//2210,2210,value.shape[1]) for key,value in Nods.items()})

    plt.plot(*reduce(add,[(Pow[:,0],Pow[:,1],'-o') for Pow in Pows.values()]))
    plt.legend(Pows.keys())

    file = 'INFO\\solaeki.reac'
    alf = np.loadtxt(file)
    plt.plot(alf[:, 0], alf[:, 1], 'o')
    return


if __name__ == '__main__':
    # alf = GroupFlux('aer.cdb.flux')
    # plt.plot(range(10),alf[0,0,:,1],'-o',range(10),alf[0,0,:,0],'o-')

    Geo = Geometry('aer_geo.ci@')
    import re
    alfred = re.compile(r'(?<=\s{2}3\n)(?:[\s0-9]{3}){4}([\s0-9]{3})')
    findings = alfred.search(open('aer_geo.ci@').read())
    pass

