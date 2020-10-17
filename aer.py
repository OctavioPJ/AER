from PyCAR.PyCIT.lector_mest_posta import LoadCDP, get_reactivity_pro as grp, AddableDict
from PyCAR.PyCIT.FT import MeshFlux
import os
from citvappru.SourceCAREM2_1_1 import Geometry, PrecCalc
from aer_construction import AerModel
import numpy as np
import re

os.chdir('D:\\AER\\')
BLACKABSORBER = 9999
RADIALREFLECTOR = 4000
AXIALREFLECTOR = 4001  # 4001 MATERIALES
DEFAULT_STATE = 0
DEFAULT_BURNUP = 0

# TODO chequear la simetría del mapeo, hay un problema respecto a como se está mapeando la fuente
#  que se puede ver cargando las bases de datos aerS.cdb(cálculo con fuente) y aer_eq.cdb(cálculo crítico)
#  el movimiento dentro de la sección 5 sería (para subir verticalmente en la grilla hexagonal)
#  (q,r) - > (q+1,r) == (x,y) -> (x+1,y+1)


class AerMap:
    """

    Esta clase sirve para identificar cada elemento combustible con su respectivo archivo *.cdp o
    los datos para neutrones retardados.
    Las clases que heredan de esta clase sirven como bases de datos, cuyos atributos son los archivos, o datos,
    relacionados respecto a dicho combustible.
    meshmaterial -> fuel element
    """

    def MapMaterial(self, meshmaterial, **kwargs):
        list_fuels = kwargs['list_fuels']
        nfuels = len(list_fuels)
        nslices = 10
        assert 'list_fuels' in kwargs, 'Falta la lista de combustibles'
        if meshmaterial <= nfuels * nslices:
            FuelNumbr = (meshmaterial - 1) % nfuels
            FuelPlane = (meshmaterial - 1) // nfuels + 1  # esto ayuda a que se vean las barras
            if list_fuels[FuelNumbr][1] < 20:  # TODOS ESTOS REPRESENTAN EECC SIN BARRAS
                attr = '_tipo{}'.format(list_fuels[FuelNumbr][1])
            else:
                # TODO: implementar un switch-case analogo para los combustibles
                assert 'time' in kwargs, 'No se añadió paso temporal'
                FuelType = list_fuels[FuelNumbr][1]
                if FuelType == 21:
                    Threshold = 1.0
                    Time = kwargs['time']
                    Insertion_Velocity = 250 / 10.0  # Hasta el fondo del nucleo en 10 segundos
                    Insertion_Slice = 8 + (Insertion_Velocity * Time) // 25 if Time > Threshold else 8
                    if FuelPlane <= Insertion_Slice:
                        attr = '_tipo4'
                    else:
                        attr = '_tipo1'
                elif FuelType == 26:
                    Time = kwargs['time']
                    Extraction_Velocity = 200 / 0.08  # 200 cm de extracción en 0.08 seg
                    Insertion = (200 - Extraction_Velocity * Time) if Time < 0.08 else 0
                    Insertion_Slice = Insertion // 25  # 25 cm de nodo
                    if FuelPlane <= Insertion_Slice:
                        attr = '_tipo4'
                    else:
                        attr = '_tipo2'
                else:  # BARRAS 23 y 25 PARA SCRAM
                    Threshold = 1.0
                    Time = kwargs['time']
                    Insertion_Velocity = 250 / 10.0  # Hasta el fondo del nucleo en 10 segundos
                    Insertion_Slice = (Insertion_Velocity * Time) // 25 if Time > Threshold else 0
                    if FuelPlane <= Insertion_Slice:
                        attr = '_tipo4'
                    else:
                        attr = '_tipo2' if FuelType == 23 else '_tipo1'
        else:
            if meshmaterial == RADIALREFLECTOR:
                attr = '_tipo5'
            elif meshmaterial == AXIALREFLECTOR:
                attr = '_tipo6'
            else:
                attr = '_NULL'
        return getattr(self, attr)

    pass  # AER_MAP


class NuFissionMap(AerMap):
    wdir = 'D:\\AER\\'

    def __init__(self, **kwargs):
        if 'wdir' in kwargs:
            setattr(self, 'wdir', kwargs['wdir'])

        wdir = getattr(self, 'wdir')
        self._tipo1 = LoadCDP('TIPO1.cdp')[0][0.0]
        self._tipo2 = LoadCDP('TIPO2.cdp')[0][0.0]
        self._tipo3 = LoadCDP('TIPO3.cdp')[0][0.0]
        self._tipo4 = LoadCDP('TIPO4.cdp')[0][0.0]
        self._tipo5 = LoadCDP('TIPO5.cdp')[0][0.0]
        self._tipo6 = LoadCDP('TIPO6.cdp')[0][0.0]
        self._NULL = {'XS': np.zeros(self._tipo1['XS'].shape),
                      'SM': np.zeros(self._tipo1['SM'].shape)}

    pass  # NuFissionMap


class KineticMap(AerMap):

    def __init__(self, **kwargs):
        v1 = 1.25E+7
        v2 = 2.50E+5
        self._tipo1 = self._tipo2 = self._tipo3 = self._tipo4 = self._tipo5 = self._tipo6 = \
            {'Beta': np.array([0.000247, 0.0013845, 0.001222, 0.0026455, 0.000832, 0.000169]),
             'Decays': np.array([0.012700, 0.0317000, 0.115000, 0.3110000, 1.400000, 3.870000]),
             'Neutron Velocity': np.array([1 / v1, 1 / v2])}

        self._NULL = {'Beta': [0] * 6, 'Decays': [0] * 6, 'Neutron Velocity': [0, 0]}

    pass  # KineticMap


class AerEvolution(AerModel):
    def __init__(self):
        super().__init__()  # llama a la creación de AerModel
        print('- Inicializando Transitorio')
        self._C0 = {}
        self.__Ct_1 = {}
        self._Ct = {}
        self.__FILENAME = 'aer.cii'
        self.__ROOTFILE = self.__FILENAME[:-4]
        self.__DATABASE = self.__ROOTFILE + '_eq.cdb'
        self.__Geom = Geometry(self.__FILENAME.replace('.cii', '_geo.ci@'))
        self.__NuFis = NuFissionMap()
        self.__KM = KineticMap()
        self._NOG = 2  # TODO: Sería mejor extraer del ci@?
        self.chi_g = [1.0, 0.0]  # TODO: Sería mejor extraer del ci@?
        self.__Flux_t_1 = self.Flux = MeshFlux(self.__DATABASE + '.meshflux', *self.__Geom.Cantidad_de_Nodos(),
                                               self._NOG)
        Nx, Ny, Nz, _NOG = self.Flux.shape[1:]
        self.BetaM = np.empty((1, Nx, Ny, Nz, 6))
        self.LambdaM = np.empty((1, Nx, Ny, Nz, 6))
        self.NuFisM = np.empty((1, Nx, Ny, Nz, _NOG))
        self.VelocityM = np.empty((1, Nx, Ny, Nz, _NOG))
        state = 0
        static_react = -123.1  # reactividad inicial inicial, no inicial
        self.keff = 1 / (1 - static_react / 100000)
        self._Q = {}
        self.EQUILIBRIUM = True
        self.powers = []

        for _x in range(Nx):
            for _y in range(Ny):
                for _z in range(Nz):
                    meshmaterial = self.__Geom.sc5[_x][_y][_z]

                    kwargs = {'list_fuels': self.list_fuels, 'time': 0}
                    self.BetaM[state][_x][_y][_z][:] = \
                        self.__KM.MapMaterial(meshmaterial, **kwargs)['Beta']
                    self.LambdaM[state][_x][_y][_z][:] = \
                        self.__KM.MapMaterial(meshmaterial, **kwargs)['Decays']
                    self.VelocityM[state][_x][_y][_z][:] = \
                        self.__KM.MapMaterial(meshmaterial, **kwargs)['Neutron Velocity']
                    self.NuFisM[state][_x][_y][_z][:] = \
                        self.__NuFis.MapMaterial(meshmaterial, **kwargs)['XS'][:, 3] / self.keff

    @property
    def Ct_1(self):
        """
        Precursores calculados en el paso anterior
        """
        return self.__Ct_1.copy()

    @property
    def Ct(self):
        """
        Precursores calculados hasta ahora.
        """
        return AddableDict(self._Ct)

    @property
    def C0(self):
        """
        Precursores de Equilibrio (reactor estático)
        """
        return AddableDict(self._C0)

    def equilibrium_precrs(self):
        print('- Calculando precursores en equilibrio')
        state = 0
        Vmesh = self.__Geom.Vmesh()

        NPRC = self.BetaM.shape[-1]
        self._C0[state] = {}
        for nx in range(self.Flux.shape[1]):
            self._C0[state][nx] = {}
            for ny in range(self.Flux.shape[2]):
                self._C0[state][nx][ny] = {}
                for nz in range(self.Flux.shape[3]):
                    FluxL = [self.Flux[state, nx, ny, nz, group] for group in range(self.Flux.shape[-1])]
                    NuFisL = [self.NuFisM[state][nx][ny][nz][group] for group in range(self.Flux.shape[-1])]
                    Nu_FluxM = [NuFisL[group] * FluxL[group] * Vmesh for group in range(self.Flux.shape[-1])]

                    Bet_k = self.BetaM[state][nx][ny][nz]
                    Lamb_k = self.LambdaM[state][nx][ny][nz]
                    Nu_Flux = sum(Nu_FluxM)

                    self._C0[state][nx][ny][nz] = [Bet_k[prec] * Nu_Flux / Lamb_k[prec]
                                                   if Lamb_k[prec] != 0 else 0.0
                                                   for prec in range(NPRC)]
        self.__Ct_1 = self._C0.copy()
        self._Ct = self._C0.copy()

    def calculate_precrs(self, dt):
        TFlux = MeshFlux(self.__DATABASE + '.meshflux', *self.__Geom.Cantidad_de_Nodos()[1:4])
        self._Ct = PrecCalc(self.__Ct_1, TFlux, self.NuFisM, self.__Geom.Vmesh(), dt
                            , LambdaM=self.LambdaM
                            , BetaM=self.BetaM)

        self.__Ct_1 = self._Ct.copy()
        self.__Flux_t_1 = TFlux
        return

    @property
    def Q_as_array(self):
        """
        Fuente calculada al instante t como array de numpy
        """
        return AddableDict(self._Q).as_array()

    def calculate_source(self, **kwargs):
        print('- Calculando la fuente')
        DERIVATIVE = kwargs['DERIVATIVE']
        if DERIVATIVE:
            dt = kwargs['dt']
        else:
            dt = 1
        Vmesh = self.__Geom.Vmesh()
        NPRC = 6
        for group in range(self.Flux.shape[-1]):
            self._Q[group] = {}
            for state in range(self.Flux.shape[0]):
                self._Q[group][state] = {}
                for nx in range(self.Flux.shape[1]):
                    self._Q[group][state][nx] = {}
                    for ny in range(self.Flux.shape[2]):
                        self._Q[group][state][nx][ny] = {}
                        for nz in range(self.Flux.shape[3]):
                            _Lmk = self.LambdaM[state][nx][ny][nz]
                            _C = self._Ct[state][nx][ny][nz]

                            _invV = self.VelocityM[state, nx, ny, nz, group]
                            T_1Flux = self.__Flux_t_1[state, nx, ny, nz, group]

                            self._Q[group][state][nx][ny][nz] = \
                                self.chi_g[group] * sum([_Lmk[prc] * _C[prc] for prc in range(NPRC)]) \
                                + (_invV / dt * T_1Flux * Vmesh if DERIVATIVE else 0)

    def move_control_rods(self, **kwargs):
        Threshold = 1.0
        assert 'time' in kwargs, 'No se ha establecido el tiempo de cálculo'
        Time = kwargs['time']
        Extraction_Velocity = 200 / 0.08
        Extraction_CR_26 = 8 - (Extraction_Velocity * Time) // 25 if Time <= 0.08 else 0
        self.InsertControlRod(Extraction_CR_26, 26)

        if Time > Threshold:
            Insertion_Velocity = 250 / 10.0  # Hasta el fondo del nucleo en 10 segundos
            Insertion_CR_21 = 8 + (Insertion_Velocity * Time) // 25 if Time <= 2.0 else 10
            Insertion_CR_SS = (Insertion_Velocity * Time) // 25 if Time <= 11.0 else 10
            # Control Rod Secutiry System (23,25)
            self.SCRAM(Insertion_CR_SS)
            self.InsertControlRod(Insertion_CR_21, 21)
            # TODO testear este método y llevarlo a archivo
        return

    def source_to_file(self):
        print('- Escribiendo el Archivo source.dat')
        _NOG = 2
        with open('source.dat', 'w') as fod:
            for group in range(_NOG):
                for state in range(self.Flux.shape[0]):
                    for nz in range(self.Flux.shape[3]):
                        for ny in range(self.Flux.shape[2]):
                            for nx in range(self.Flux.shape[1]):
                                fod.write('{:15.7E}'.format(self._Q[group][state][nx][ny][nz]))
                            fod.write('\n')
                fod.write('\n')

    def Execute(self,**kwargs):
        self.EQUILIBRIUM = False
        print('- Ejecuntando CITVAP')
        DERIVATIVE = kwargs['DERIVATIVE']
        if DERIVATIVE:
            dt = kwargs['dt']
        else:
            dt = 1
        time = kwargs['time']
        p = re.compile(r'POWER\(WATTS\)\s+([0-9]\.[0-9]{5}E\+[0-9]{2})')
        Nx, Ny, Nz = self.__Geom.Cantidad_de_Nodos()
        OsArgs = (self.__FILENAME.replace('.cii', 'S.cii'),
                  self._NOG, Nx, Ny, Nz, self.EQUILIBRIUM,
                  *['{:12.5E}'.format(_d) for _d in [dt * DERIVATIVE, time]])
        # PARA HACER UNA PERTURBACION ESCALOR,COLOCAR t=1
        # PARA CALCULAR SIN DERIVADA TEMPORAL dt=0
        try:
            os.system('ExEqAer.bat ' + ' '.join(map(str, OsArgs)))
            fid = open(self.__ROOTFILE + 'S.cio', 'r')
            self.powers.append(float(next(p.finditer(fid.read())).groups()[0]))
            print(self.powers[-1])
            fid.close()
            if self.EQUILIBRIUM: self.EQUILIBRIUM = False
        except StopIteration:
            FAILED_COMPLETE_TEST = True
            pass

    # TODO: traer el script a python para ser ejecutado con subprocess.Popen
    pass  # AerEvolution


def mainnomain():
    os.chdir('D:\\AER\\')
    FILENAME = 'aer.cii'
    ROOTFILE = FILENAME[:-4]
    DATABASE = ROOTFILE + '_eq.cdb'

    Geom = Geometry(FILENAME.replace('.cii', '_geo.ci@'))
    NuFis = NuFissionMap()
    KM = KineticMap()
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

                KinParam = KM.MapMaterial(meshmaterial)  # Obsoleto

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
                                + _invV / dt * T_1Flux * Vmesh * DERIVATIVE  #

        with open('source.dat', 'w') as fod:
            for group in range(_NOG):
                for state in range(Flux.shape[0]):
                    for nz in range(Flux.shape[3]):
                        for ny in range(Flux.shape[2]):
                            for nx in range(Flux.shape[1]):
                                fod.write('{:15.7E}'.format(Q[group][state][nx][ny][nz]))
                fod.write('\n')
        # Ejecución de archivo AQUI
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
    return

def Main():
    aerT = AerEvolution()
    aerT.equilibrium_precrs()
    aerT.calculate_source(DERIVATIVE=False)
    aerT.source_to_file()
    aerT.Execute(DERIVATIVE=False,time=0.0)
    return


if __name__ == '__main__':
    # NuMap = NuFissionMap()
    # for meshmaterial in range(1,350):
    #     NuMap.MapMaterial(meshmaterial)
    aerT = AerEvolution()
    pass
