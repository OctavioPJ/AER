from PyCAR.PyCIT.lector_mest_posta import LoadCDP, get_reactivity_pro as grp
from PyCAR.PyCIT.FT import MeshFlux
import os
from citvappru.SourceCAREM2_1_1 import Geometry, PrecCalc
import numpy as np
import re

BLACKABSORBER = 10
DEFAULT_STATE = 0
DEFAULT_BURNUP = 0


class AER_MAP(object):
    """

    meshmaterial -> fuel element
    Esta clase sirve para identificar cada elemento combustible con su respectivo archivo *.cdp o
    los datos para neutrones retardados.
    Las clases que heredan de esta clase sirven como bases de datos, cuyos atributos son los archivos, o datos,
    relacionados respecto a dicho combustible.
    """

    def MapMaterial(self, meshmaterial, **kwargs):
        attr = '_tipo' + str(meshmaterial)
        try:
            return getattr(self, attr)[0][0.0]
        except AttributeError as err:
            return getattr(self, '_NULL')


class Nu_Fission_Map(AER_MAP):
    wdir = 'D:\\AER\\'

    def __init__(self, **kwargs):
        if 'wdir' in kwargs:
            setattr(self, 'wdir', kwargs['wdir'])

        wdir = getattr(self, 'wdir')
        for k in range(1, 7):
            setattr(self, '_tipo{0}'.format(k), LoadCDP(wdir + 'TIPO{}.cdp'.format(k)))

        self._NULL = {'XS': np.zeros(self._tipo1[0][0.0]['XS'].shape),
                      'SM': np.zeros(self._tipo1[0][0.0]['SM'].shape)}


class Kinetic_Map(AER_MAP):

    def __init__(self, **kwargs):
        v1 = 1.25E+7
        v2 = 2.50E+5
        self._tipo1 = self._tipo2 = self._tipo3 = self._tipo4 = self._tipo5 = self._tipo6 = \
            {0: {0.0: {'Beta': np.array([0.000247, 0.0013845, 0.001222, 0.0026455, 0.000832, 0.000169]),
                       'Decays': np.array([0.012700, 0.0317000, 0.115000, 0.3110000, 1.400000, 3.870000]),
                       'Neutron Velocity': np.array([1 / v1, 1 / v2])}}}

        self._NULL = {'Beta': [0] * 6, 'Decays': [0] * 6, 'Neutron Velocity': [0, 0]}


if __name__ == '__main__':
    os.chdir('D:\\AER\\')
    FILENAME = 'aer.cii'
    ROOTFILE = FILENAME[:-4]
    DATABASE = ROOTFILE + '_eq.cdb'

    Geom = Geometry(FILENAME.replace('.cii', '_geo.ci@'))
    NuFis = Nu_Fission_Map()
    KM = Kinetic_Map()
    _NOG = 2
    Flux = MeshFlux(DATABASE + '.meshflux', *Geom.Cantidad_de_Nodos(), _NOG)
    Nx, Ny, Nz = Flux.shape[1:4]

    BetaM = np.empty((1, Nx, Ny, Nz, 1))
    LambdaM = np.empty((1, Nx, Ny, Nz, 1))
    NuFisM = np.empty((1, Nx, Ny, Nz, _NOG))
    VelocityM = np.empty((1, Nx, Ny, Nz, _NOG))

    state = 0

    Vmesh = Geom.Vmesh()

    react = -123.1  # reactividad inicial inicial, no inicial
    keff = 1 / (1 - react / 100000)

    for _x in range(Nx):
        for _y in range(Ny):
            for _z in range(Nz):
                meshmaterial = Geom.sc5[_x][_y][_z]

                KinParam = KM.MapMaterial(meshmaterial)

                BetaM[state][_x][_y][_z][:] = KinParam['Beta']

                LambdaM[state][_x][_y][_z][:] = KinParam['Decays']

                VelocityM[state][_x][_y][_z][:] = KinParam['Neutron Velocity']

                NuFisM[state][_x][_y][_z][:] = NuFis.MapMaterial(meshmaterial)['XS'][:, 3] / keff

    C0 = {}
    NPRC = BetaM.shape[-1]
    C0[state] = {}
    for nx in range(Flux.shape[1]):
        C0[state][nx] = {}
        for ny in range(Flux.shape[2]):
            C0[state][nx][ny] = {}
            for nz in range(Flux.shape[3]):
                FluxL = [Flux[state, nx, ny, nz, group] for group in range(Flux.shape[-1])]
                NuFisL = [NuFisM[state][nx][ny][nz][group] for group in range(Flux.shape[-1])]
                Nu_FluxM = [NuFisL[group] * FluxL[group] * Vmesh for group in range(Flux.shape[-1])]

                Bet_k = BetaM[state][nx][ny][nz]
                Lamb_k = LambdaM[state][nx][ny][nz]
                Nu_Flux = sum(Nu_FluxM)

                C0[state][nx][ny][nz] = [Bet_k[prec] * Nu_Flux / Lamb_k[prec]
                                         if Lamb_k[prec] != 0 else 0.0
                                         for prec in range(NPRC)]

    p = re.compile(r'POWER\(WATTS\)\s+([0-9]\.[0-9]{5}E\+[0-9]{2})')
    powers = []

    Q = {}
    EQUILIBRIUM = True
    chi_g = [1.0, 0.0]

    v1 = 1.25E+7
    v2 = 2.50E+5
    dt = 0.01
    tfinal = 0.5
    DATABASE = ROOTFILE + 'S.cdb'
    Times = np.arange(0, tfinal + 2 * dt, dt)
    FAILED_COMPLETE_TEST = False
    DERIVATIVE = True
    for t in Times:

        if EQUILIBRIUM:
            C_t_1 = C0.copy()
            C_t = C0.copy()
            Flux_t_1 = Flux
        else:
            TFlux = MeshFlux(DATABASE + '.meshflux', Nx, Ny, Nz, _NOG)
            # noinspection PyUnboundLocalVariable
            C_t = PrecCalc(C_t_1, TFlux, NuFisM, Vmesh, dt
                           , LambdaM=LambdaM
                           , BetaM=BetaM)

            C_t_1 = C_t.copy()
            Flux_t_1 = TFlux

        for group in range(Flux.shape[-1]):
            Q[group] = {}
            for state in range(Flux.shape[0]):
                Q[group][state] = {}
                for nx in range(Flux.shape[1]):
                    Q[group][state][nx] = {}
                    for ny in range(Flux.shape[2]):
                        Q[group][state][nx][ny] = {}
                        for nz in range(Flux.shape[3]):
                            _Lmk = LambdaM[state][nx][ny][nz]
                            _C = C_t[state][nx][ny][nz]

                            _invV = VelocityM[state, nx, ny, nz, group]
                            T_1Flux = Flux_t_1[state, nx, ny, nz, group]

                            Q[group][state][nx][ny][nz] = \
                                chi_g[group] * sum([_Lmk[prc] * _C[prc] for prc in range(NPRC)]) \
                                + _invV / dt * T_1Flux * Vmesh * DERIVATIVE

        with open('source.dat', 'w') as fod:
            for group in range(_NOG):
                for state in range(Flux.shape[0]):
                    for nz in range(Flux.shape[3]):
                        for ny in range(Flux.shape[2]):
                            for nx in range(Flux.shape[1]):
                                fod.write('{:15.7E}'.format(Q[group][state][nx][ny][nz]))
                fod.write('\n')

        OsArgs = (FILENAME.replace('.cii', 'S.cii'),
                  _NOG, Nx, Ny, Nz, EQUILIBRIUM,
                  *['{:12.5E}'.format(_d) for _d in [dt * DERIVATIVE, t]])
        # PARA HACER UNA PERTURBACION ESCALOR,COLOCAR t=1
        # PARA CALCULAR SIN DERIVADA TEMPORAL dt=0
        try:
            os.system('ExAer.bat ' + ' '.join(map(str, OsArgs)))
            fid = open(ROOTFILE + 'S.cio', 'r')
            powers.append(float(next(p.finditer(fid.read())).groups()[0]))
            print(powers[-1])
            fid.close()
            if EQUILIBRIUM: EQUILIBRIUM = False
        except StopIteration:
            FAILED_COMPLETE_TEST = True
            break
            pass

    Normalized_Pow = np.array(powers) / 1E+06
    if FAILED_COMPLETE_TEST: Times = [k * dt for k in range(len(powers))]
