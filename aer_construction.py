from aer_fuels import MATERIAL, CONTROL_ROD, FUEL_ELEMENT, SECCION8
import sys
from itertools import count, groupby
from abc import abstractmethod,ABC
from copy import deepcopy

ABSORBER = 9999
REFLECTOR = 5

sys.path.extend('D:\\aer1')


class NoneExistanteHalf(Exception):
    pass

# LA IDEA PARA LA IMPLEMENTACIÓN DE LAS CLASES CORE Y HEXAGON FUERON TOMADAS DE
# https://www.redblobgames.com/grids/hexagons/#map-storage

class Triangle(object):
    def __init__(self,*args):
        self._mat = args[0]
        self._dir = args[1]  # ^ o v

        self.neighbour_30 = self.neighbour_90 = self.neighbour_150 = \
            self.neighbour_210 = self.neighbour_270 = self.neighbour_330 = None

    @property
    def string(self):
        triang_string = '{:4d}'.format(self._mat)
        if self._dir == 'upper' and self.neighbour_30 is not None:
            triang_string += self.neighbour_30.string
        elif self._dir == 'lower' and self.neighbour_330 is not None:
            triang_string += self.neighbour_330.string
        else:
            triang_string += ' /\n'
        return triang_string

    # TODO Chequear si es necesaria esta conversión
    #  y, en caso de serlo, terminarla.
    #  la implementación de esta simetría sigue hasta aer.py

    pass  # Triangle


class Hexagon(object):
    def __init__(self, *args,**kwargs):
        if args:
            if args[0] != REFLECTOR and args[0] != ABSORBER\
                    and 'COUNTER' in kwargs:
                self.__material = - next(kwargs['COUNTER'])
                self.__TYPE__ = args[0]
            else:
                self.__TYPE__ = self.__material = args[0]

        else:
            self.__TYPE__ = self.__material = ABSORBER

        self.__upper_half = [self.__material] * 3  # [M1,M2,M3]
        #
        #                 UPPER HALF
        #                 ----------
        #                / \      / \
        #               /   \ M2 /   \
        #              / M1  \  /  M3 \
        #             /_______\/_______\
        #
        self.__lower_half = [self.__material] * 3  # [M1,M2,M3]
        #
        #                  LOWER HALF
        #             -------------------  
        #              \      / \       /
        #               \ M1 /   \  M3 /
        #                \  /  M2 \   /
        #                 \/_______\ /
        #
        self.FULL = True
        if len(args) > 1:
            if args[1] == 'upper':
                self.__lower_half = None

            elif args[1] == 'lower':
                self.__upper_half = None
            self.FULL = False
        # VECINOS (NEIGHBOUR) SEGUN SU ANGULO
        #
        #                \   SELF.N*90   /
        #                 \             /
        #      SELF.N*150  ------------      SELF.N*30
        #                 /             \
        #                /               \
        #       _______ /       SELF      \ __________
        #               \                 /
        #                \               /
        #                 \             /
        #      SELF.N*210   ------------     SELF.N*330
        #                 /              \
        #                /    SELF.N*270  \
        #
        self.neighbour_30 = self.neighbour_90 = self.neighbour_150 = \
            self.neighbour_210 = self.neighbour_270 = self.neighbour_330 = None

        # COORDINATES
        self.q = self.r = None
        self.x = self.y = self.z = None
        pass

    @property
    def type(self):
        if self.__material not in [REFLECTOR, ABSORBER]:
            return -self.__material, self.__TYPE__
        return self.__material, self.__TYPE__

    def string_upper(self):
        final_string = ' '.join(map(lambda a: '{:4d}'.format(a), self.__upper_half))
        if self.neighbour_30 is not None:
            final_string += ' ' + self.neighbour_30.string_lower()
        else:
            final_string += ' /\n'
        return final_string

    def string_lower(self):
        final_string = ' '.join(map(lambda a: '{:4d}'.format(a), self.__lower_half))
        if self.neighbour_330 is not None:
            final_string += ' ' + self.neighbour_330.string_upper()
        else:
            final_string += ' /\n'
        return final_string

    @property
    def string(self):
        if self.neighbour_270 is None:
            return self.string_upper() +\
                   self.string_lower()
        return self.string_upper() +\
               self.string_lower() +\
               self.neighbour_270.string

    def add_neighbour(self, other, *args):
        angle = args[0]
        assert isinstance(other, Hexagon)
        try:
            NeighbourhoodInfo = {
                30: (self.q+1,self.r-1),
                90: (self.q  ,self.r-1),
                150: (self.q-1,self.r),
                210: (self.q-1,self.r+1),
                270: (self.q  ,self.r+1),
                330: (self.q+1,self.r)
            }
        except TypeError as err:
            if not self.q or not self.r:
                other.add_neighbour(self,(angle + 180) % 360)
                return
            else:
                raise err
        try:
            other.q,other.r = NeighbourhoodInfo[angle]
        except KeyError:
            raise ValueError('El ángulo no se encuentra\nUsar '+' '.join(map(str,NeighbourhoodInfo.keys())))
        setattr(self, 'neighbour_{}'.format(angle), other)
        setattr(other, 'neighbour_{}'.format((angle + 180) % 360), self)
        return

    def corte_inf_disponible(self):
        if self.neighbour_330 is None:
            return True
        else:
            if self.neighbour_330.type[1] == ABSORBER:
                if self.neighbour_270 is not None and self.neighbour_270.neighbour_330 is not None:
                    return self.neighbour_270.neighbour_330.corte_inf_disponible()
                return True
        return False

    def corte_inferior(self):
        # corte
        if self.__upper_half:
            self.__upper_half = [self.__material]
        if self.__lower_half:
            self.__lower_half = [self.__material, self.__material]

        self.neighbour_30 = self.neighbour_90 = None
        if self.neighbour_330 is not None:
            self.neighbour_330 = None
            self.neighbour_270.neighbour_30 = None  # vecino del vecino
            if self.neighbour_270.neighbour_330 is not None:
                self.neighbour_270.neighbour_330.corte_inferior()

    def empty(self):
        if self.__upper_half:
            #            self.__upper_half = None
            self.__upper_half = [0] * 3
        if self.__lower_half:
            #            self.__lower_half = None
            self.__lower_half = [0] * 3
        if self.neighbour_270 is not None:  self.neighbour_270.empty()

    def corte_sup_disponible(self):
        if self.neighbour_270 is None:
            return True
        else:
            if self.neighbour_270.type[1] == ABSORBER:
                if self.neighbour_330 is not None and self.neighbour_330.neighbour_270 is not None:
                    return self.neighbour_330.neighbour_270.corte_sup_disponible()
                return True
        return False

    def corte_superior(self):

        if self.__upper_half:
            self.__upper_half = [0, self.__material, self.__material]
        if self.__lower_half:
            self.__lower_half = [0,               0, self.__material]
        #
        #        if self.__upper_half:
        #            self.__upper_half = [self.__material,self.__material]
        #        if self.__lower_half:
        #            self.__lower_half = [self.__material]

        self.neighbour_150 = self.neighbour_210 = None

        if self.neighbour_330 is not None:
            self.neighbour_270.empty()
            self.neighbour_330._210n = None  # vecino del vecino

            if self.neighbour_330.neighbour_270 is not None:
                self.neighbour_330.neighbour_270.corte_superior()

    def up_material(self):
        Mats = self.__upper_half.copy() if self.__upper_half is not None else []
        if self.neighbour_30 is not None:
            Mats += self.neighbour_30.low_material()
        return Mats

    def low_material(self):
        Mats = self.__lower_half.copy() if self.__lower_half is not None else []
        if self.neighbour_330 is not None:
            Mats += self.neighbour_330.up_material()
        return Mats

    @property
    def material(self):
        if self.neighbour_270 is None:
            return [self.up_material(), self.low_material()]
        return [self.up_material(), self.low_material()] + self.neighbour_270.material

    pass  # Hexagon


class Core:
    def __init__(self, *args):
        """

        :param args:
        """
        self.__COUNTER__ = count(1)
        self._Nrows = args[0]
        self.__q = -1
        self.__r = -1
        self.__grid = []
        self.__CENTER = None
        pass

    @property
    def grid(self):
        return deepcopy(self.__grid)

    def add_ring(self,**kwargs):
        if self.__q == -1:
            # first initialize
            self.__r = self.__q = 0

            assert 'materials' in kwargs
            material = kwargs['materials']
            self.__CENTER = Hexagon(material)
            self.__grid.append([self.__CENTER])
        else:
            ring = self.__q
            self.__q += 1
            assert len(kwargs['material']) == 6 * self.__q, 'No se han insertado la cantidad suficiente de materiales'
            self.__r = len(kwargs['material'])//6
            assert isinstance(self.__grid[0],Hexagon)
            Ring = [Hexagon(material) for material in kwargs['material']]
            angle = 30
            # TODO es un poco mas complicado que esto.... haría falta algo similar a lo anterior
            for Hex in Ring:
                for k in range(self.__r):
                    self.__grid[ring][k].add_neighbour(Hex,angle)
                angle += 60
            pass
        return

    def add_column(self, **kwargs):  # agrega a la derecha
        if self.__q == -1:
            print('Initial Column')
            #            first initialize
            q = self.__q = 0

            start = 0
            stop = self._Nrows

            if 'materials' in kwargs:
                RowHash = {r: Hexagon(material, COUNTER=self.__COUNTER__)
                           for material, r in zip(kwargs['materials'], range(start, stop))}
                self.__grid.append(RowHash)
            else:
                RowHash = {r: Hexagon() for r in range(start, stop)}
                self.__grid.append(RowHash)

            self.__r = len(self.__grid[q])  # = self._Nrows

            self.__grid[0][0].q = 0
            self.__grid[0][0].r = 0

            for r in range(self._Nrows - 1):
                self.__grid[q][r].add_neighbour(self.__grid[q][r + 1], 270)

        else:
            start = self.__r - self._Nrows - 1
            stop = self.__r - 1
            print(start)
            print(stop)
            if not self.__q & 1:
                print('Even Columns, length {}'.format(self._Nrows-1)+' + 2 divided Hexs')
                #                adding even column, divided hexagons with upper_half and lower_half
                if 'materials' in kwargs:
                    assert len(kwargs['materials']) == self._Nrows - 1
                    RowHash = {start: Hexagon(ABSORBER, 'lower')}
                    RowHash.update({r : Hexagon(material,COUNTER=self.__COUNTER__)
                                    for material, r in zip(kwargs['materials'], range(start+1,stop))})
                    RowHash.update({stop :Hexagon(ABSORBER, 'upper')})
                    self.__grid.append(RowHash)
                else:
                    RowHash = {start: Hexagon(ABSORBER, 'lower')}
                    RowHash.update({r : Hexagon() for r in range(start+1,stop)})
                    RowHash.update({stop :Hexagon(ABSORBER, 'upper')})
                    self.__grid.append(RowHash)

                q = self.__q + 1
                for r in range(start,stop):
                    self.__grid[q - 1][r + 1].add_neighbour(self.__grid[q][r + 1],
                                                    330)  # acopla abajo a la derecha con la izquierda
                    self.__grid[q][r].add_neighbour(self.__grid[q - 1][r + 1],
                                                    210)  # acopla abajo a la izquierda con la derecha
                    self.__grid[q][r].add_neighbour(self.__grid[q][r + 1],270)  # acopla con el de abajo

            else:
                #                adding odd column,full hexagons
                print('Odd Columns, length {}'.format(self._Nrows))
                if 'materials' in kwargs:
                    assert len(kwargs['materials']) == self._Nrows
                    RowHash = { r :Hexagon(material, COUNTER=self.__COUNTER__)
                                for material, r in zip(kwargs['materials'], range(start,stop))}
                    self.__grid.append(RowHash)
                else:
                    RowHash = {r: Hexagon() for r in range(start, stop)}
                    self.__grid.append(RowHash)

                q = self.__q + 1
                for r in range(start,stop):
                    self.__grid[q - 1][r].add_neighbour(self.__grid[q][r],330)  # acopla abajo a la derecha con la izquierda
                    self.__grid[q][r].add_neighbour(self.__grid[q - 1][r + 1],210)  # acopla abajo a la izquierda con la derecha

                for r in range(start,stop-1):
                    self.__grid[q][r].add_neighbour(self.__grid[q][r + 1], 270)  # acopla con el de abajo

                self.__r -= 1

            self.__q += 1
        return

    @property
    def string(self):
        assert isinstance(self.__grid[0][0],Hexagon)
        return self.__grid[0][0].string

    def cortar_para_sc5(self):
        for sub_grid in self.__grid:
            if sub_grid[min(sub_grid)].corte_inf_disponible():
                sub_grid[min(sub_grid)].neighbour_330.corte_inferior()
                sub_grid[min(sub_grid)].neighbour_30 = None
                break

        for r,hexag in self.__grid[0].items():
            if hexag.corte_sup_disponible():
                hexag.neighbour_270.corte_superior()
                break
        return

    def filled_to_string(self):
        """
        esto nunca andubo
        """
        max_msize = max([len(sub) for sub in self.__grid[0][0].material])

        MESHED_GRID = []
        APPEND_AT_THE_END = False

        for sub in self.__grid[0][0].material:
            if len(sub) == max_msize:
                APPEND_AT_THE_END = True

            if not APPEND_AT_THE_END:
                MESHED_GRID.append([ABSORBER] * (max_msize - len(sub)) + sub)
            else:
                MESHED_GRID.append(sub + [ABSORBER] * (max_msize - len(sub)))

        #
        return ' /\n'.join(
            map(lambda sub_grid:
                ' '.join(map(lambda a: '{:4d}'.format(a)
                             , sub_grid[1])),
                enumerate(MESHED_GRID)))

    @property
    def list_materials(self):
        return [hexag.type for sub_grid in self.__grid
                for r,hexag in sub_grid.items()]

    @property
    def list_fuels(self):
        return [hexag.type for sub_grid in self.__grid
                for r,hexag in sub_grid.items()
                if hexag.type[1] not in [ABSORBER, REFLECTOR]]

    @property
    def mesh_materials(self):
        return self.__grid[0][0].material

    @abstractmethod
    def InsertControlRod(self, *args):
        pass

    pass  # Core


class AerModel(Core):
    _APOTHEM_ = 14.7
    # TODO: Implementar geometría al modelo (pasandolo a sección 3,4 y 5)
    #    sin necesidad de pasar por citation

    def __init__(self):
        super().__init__(22)
        self._mat_count = count(0)
        self._mat = {}
        #    LINEA MAS LARGA 22
        BA = ABSORBER
        # self = Core(22)
        #    linea mas delgada 21
        self.add_column()  # COLUMNA VACIA, SOLAMENTE ABSORBENTE NEGRO                                                     <-------------------Y
        self.add_column(materials=  [ BA, BA, BA, BA, BA, BA, BA, BA,  5,  5,  5,  5,  5, BA, BA, BA, BA, BA, BA, BA, BA])  # 5 ####           X
        self.add_column(materials=[ BA, BA, BA, BA, BA, BA,  5,  5,  5,  3,  3,  3,  3,  5,  5,  5, BA, BA, BA, BA, BA, BA])  # 10####         |
        self.add_column(materials=  [ BA, BA, BA, BA,  5,  5,  3,  3,  3,  3,  2,  3,  3,  3,  3,  5,  5, BA, BA, BA, BA])  # 13####           |
        self.add_column(materials=[ BA, BA, BA, BA,  5,  3, 23,  3,  3, 25,  2,  2, 25,  3,  3, 23,  3,  5, BA, BA, BA, BA])  # 14####         |
        self.add_column(materials=  [ BA, BA,  5,  5,  3,  3,  3,  1,  2,  1,  2,  1,  2,  1,  3,  3,  3,  5,  5, BA, BA])  # 17####           |
        self.add_column(materials=[ BA, BA,  5,  3,  3,  3,  1,  2,  1,  2,  1,  1,  2,  1,  2,  1,  3,  3,  3,  5, BA, BA])  # 18####         |
        self.add_column(materials=  [ BA,  5,  3,  3, 25,  2,  1, 21,  2,  1, 23,  1,  2, 21,  1,  2, 25,  3,  3,  5, BA])  # 19####           |
        self.add_column(materials=[ BA,  5,  3,  2,  2,  1,  2,  2,  1,  2,  1,  1,  2,  1,  2,  2,  1,  2,  2,  3,  5, BA])  # 20####         |
        self.add_column(materials=  [  5,  3,  3,  2,  2,  1,  1,  2,  1,  2,  2,  2,  1,  2,  1,  1,  2,  2,  3,  3,  5])  # 21####           |
        self.add_column(materials=[ BA,  5,  3, 25,  1,  1, 23,  1,  2, 23,  1,  1, 23,  2,  1, 23,  1,  1, 25,  3,  5, BA])  # 20####         |
        self.add_column(materials=  [  5,  3,  3,  2,  2,  1,  1,  2,  1,  2,  2,  2,  1,  2,  1,  1,  2,  2,  3,  3,  5])  # 21####           v
        self.add_column(materials=[  5,  3,  3,  1,  1,  2,  2,  2,  1,  2,  1,  1,  2,  1,  2,  2,  2,  1,  1,  3,  3,  5])  # 22####         
        self.add_column(materials=  [  5, 23,  3,  2, 26,  1,  1, 23,  2,  1, 21,  1,  2, 23,  1,  1, 21,  2,  3, 23,  5])  # 21----MITAD DEL NUCLEO
        self.add_column(materials=[  5,  3,  3,  1,  1,  2,  2,  2,  1,  2,  1,  1,  2,  1,  2,  2,  2,  1,  1,  3,  3,  5])  # 22####
        self.add_column(materials=  [  5,  3,  3,  2,  2,  1,  1,  2,  1,  2,  2,  2,  1,  2,  1,  1,  2,  2,  3,  3,  5])  # 21####
        self.add_column(materials=[ BA,  5,  3, 25,  1,  1, 23,  1,  2, 23,  1,  1, 23,  2,  1, 23,  1,  1, 25,  3,  5, BA])  # 20####
        self.add_column(materials=  [  5,  3,  3,  2,  2,  1,  1,  2,  1,  2,  2,  2,  1,  2,  1,  1,  2,  2,  3,  3,  5])  # 21####
        self.add_column(materials=[ BA,  5,  3,  2,  2,  1,  2,  2,  1,  2,  1,  1,  2,  1,  2,  2,  1,  2,  2,  3,  5, BA])  # 20####
        self.add_column(materials=  [ BA,  5,  3,  3, 25,  2,  1, 21,  2,  1, 23,  1,  2, 21,  1,  2, 25,  3,  3,  5, BA])  # 19####
        self.add_column(materials=[ BA, BA,  5,  3,  3,  3,  1,  2,  1,  2,  1,  1,  2,  1,  2,  1,  3,  3,  3,  5, BA, BA])  # 18####
        self.add_column(materials=  [ BA, BA,  5,  5,  3,  3,  3,  1,  2,  1,  2,  1,  2,  1,  3,  3,  3,  5,  5, BA, BA])  # 17####
        self.add_column(materials=[ BA, BA, BA, BA,  5,  3, 23,  3,  3, 25,  2,  2, 25,  3,  3, 23,  3,  5, BA, BA, BA, BA])  # 14####
        self.add_column(materials=  [ BA, BA, BA, BA,  5,  5,  3,  3,  3,  3,  2,  3,  3,  3,  3,  5,  5, BA, BA, BA, BA])  # 13####
        self.add_column(materials=[ BA, BA, BA, BA, BA, BA,  5,  5,  5,  3,  3,  3,  3,  5,  5,  5, BA, BA, BA, BA, BA, BA])  # 10####
        self.add_column(materials=  [ BA, BA, BA, BA, BA, BA, BA, BA,  5,  5,  5,  5,  5, BA, BA, BA, BA, BA, BA, BA, BA])  # 5 ####
        self.add_column()  # COLUMNA VACIA, SOLAMENTE ABSORBENTE NEGRO

        self._sc8 = SECCION8('aer')
        for FUEL, TYPE in filter(lambda FE: FE[1] < 20, self.list_fuels):
            self._sc8.AddMaterial(FUEL_ELEMENT(FUEL, KEY='TIPO0{}'.format(TYPE), AXIAL=10))
            self._mat['FUEL ELEMENT {}'.format(FUEL)] = next(self._mat_count)

        # self._sc8.AddMaterial(CONTROL_ROD(
        self._sc8.AddMaterial(FUEL_ELEMENT(
            *tuple(
                map(lambda a: a[0],
                    filter(lambda FE: FE[1] == 21, self.list_fuels)
                    )
            )
            , KEY='TIPO02', IN='TIPO04', AXIAL=10)
        )
        self._mat['CONTROL ROD 21'] = next(self._mat_count)

        # self._sc8.AddMaterial(CONTROL_ROD(
        self._sc8.AddMaterial(FUEL_ELEMENT(
            *tuple(
                map(lambda a: a[0],
                    filter(lambda FE: FE[1] == 23, self.list_fuels)
                    )
            )
            , KEY='TIPO02', IN='TIPO04', AXIAL=10)
        )
        self._mat['CONTROL ROD 23'] = next(self._mat_count)

        # self._sc8.AddMaterial(CONTROL_ROD(
        self._sc8.AddMaterial(FUEL_ELEMENT(
            *tuple(
                map(lambda a: a[0],
                    filter(lambda FE: FE[1] == 25, self.list_fuels)
                    )
            )
            , KEY='TIPO01', IN='TIPO04', AXIAL=10)
        )
        self._mat['CONTROL ROD 25'] = next(self._mat_count)

        # self._sc8.AddMaterial(CONTROL_ROD(
        self._sc8.AddMaterial(FUEL_ELEMENT(
            *tuple(
                map(lambda a: a[0],
                    filter(lambda FE: FE[1] == 26, self.list_fuels)
                    )
            )
            , KEY='TIPO02', IN='TIPO04', AXIAL=10)
        )
        self._mat['CONTROL ROD 26'] = next(self._mat_count)

        self._sc8.AddMaterial(MATERIAL(4000, KEY='TIPO05'))
        self._mat['MATERIAL 5'] = next(self._mat_count)
        self._sc8.AddMaterial(MATERIAL(4001, KEY='TIPO06'))
        self._mat['MATERIAL 6'] = next(self._mat_count)
        pass

    def to_file(self, file):
        with open(file, 'w') as fed:  fed.write(self.string)

    def to_file_sc8(self, file):
        self._sc8.to_file(file)

    @property
    def mat(self):
        """
        Indice correspondiente a la lista self.SECCIO8.__material
        Usado para la función move_control_rod
        """
        return self._mat

    def to_file_sc5(self, file):
        self.cortar_para_sc5()
        with open(file, 'w') as fod:  fod.write(self.string)

    def InsertControlRod(self, *args):
        """
        InsertControlRod(Insertion, ControlRodNumber)

        Mueve la barra de control dentro de la sección 8

        Parameters
        ----------
        Insertion : Cantidad de trozos de combustibles a ser insertados

        ControlRodNumber : ID para barras de control (21, 23, 25 o 26)
        """
        Insertion, ControlRodNumber = args
        MatIndx = self._mat['CONTROL ROD {}'.format(ControlRodNumber)]
        self._sc8.ChangeMaterial('TIPO04',0,Insertion,MatIndx)
        return

    def SCRAM(self,*args):
        Insertion, = args
        MatIndx = map(lambda CR:self._mat['CONTROL ROD {}'.format(CR)],[23,25])
        self._sc8.ChangeMaterial('TIPO4',0,Insertion,*MatIndx)
        return
    pass  # AerModel

def main():
    import re
    alfred = re.compile(
        'TYPE \"[1-6]\"\n.*([0-9]\.[0-9]{5}E[+-][0-9]{2})\n' + '(?:([0-9]\.[0-9]{5}E[+-][0-9]{2})[\t\n])' * 6)

    cdps_re = re.compile("\* DIFFUSION   TRANSPORT   ABSORPTION  NU-FISSION   FISSION   ENERGY-FIS\.\n" +
                         ("\s+([0-9]\.[0-9]{5}E[+-][0-9]{2})" * 6 + '\n') * 2 +
                         "\* SCATTERING MATRIX\n" + ("\s([0-9]\.[0-9]{5}E[+-][0-9]{2})" * 2 + "\n") * 2)

    found = alfred.findall(open('data.dat').read())
    for k in range(2, 7):
        with open('TIPO{}or.cdp'.format(k)) as fid:
            file_string = fid.read()
            out_str = file_string
            D1, D2, Sa1, Sa2, S12, nF1, nF2 = found[k - 2]
            cdpfound = cdps_re.findall(file_string)
            out_str = out_str.replace(cdpfound[0][0], D1, 1)
            out_str = out_str.replace(cdpfound[0][1], '{:1.5E}'.format(1 / (3 * float(D1))), 1)
            out_str = out_str.replace(cdpfound[0][2], Sa1, 1)
            out_str = out_str.replace(cdpfound[0][3], nF1, 1)
            out_str = out_str.replace(cdpfound[0][4], '{:1.5E}'.format(float(nF1) / 2.55), 1)

            out_str = out_str.replace(cdpfound[0][6], D2, 1)
            out_str = out_str.replace(cdpfound[0][7], '{:1.5E}'.format(1 / (3 * float(D2))), 1)
            out_str = out_str.replace(cdpfound[0][8], Sa2, 1)
            out_str = out_str.replace(cdpfound[0][9], nF2, 1)
            out_str = out_str.replace(cdpfound[0][10], '{:1.5E}'.format(float(nF2) / 2.43), 1)

            out_str = out_str.replace(cdpfound[0][13], S12, 1)
            with open('TIPO{}.cdp'.format(k), 'w') as fod: fod.write(out_str)
    return

def test_sc4():
    with open('aer.aux') as fid:
        fstring = ''
        for line in fid.readlines():
            for group,items in groupby(line.strip('/\n').split()):
                fstring += '{:3d} * '.format(len(list(items)))+group
            fstring += ' /\n'
    with open('aer.aux.out','w') as fod:
        fod.write(fstring)
    return

def test_sc8():
    aer = AerModel()
    aer.SCRAM(5)
    aer.InsertControlRod(2, 26)
    aer.to_file_sc8('aer_out_construct.mat')
    return

def test_construct():
    core = Core(22)
    core.add_column()
    BA = 9999
    core.add_column(materials=  [ 10, BA, BA, BA, BA, BA, BA, BA,  5,  5,  5,  5,  5, BA, BA, BA, BA, BA, BA, BA, 10])  # 10####         |
    core.add_column(materials=[  5, BA, BA, BA, BA, BA,  5,  5,  5,  3,  3,  3,  3,  5,  5,  5, BA, BA, BA, BA, BA,  5])  # 10####         |
    core.add_column(materials=[BA, BA, BA, BA, 5, 5, 3, 3, 3, 3, 2, 3, 3, 3, 3, 5, 5, BA, BA, BA, BA])  # 13####           |
    core.add_column(materials=[BA, BA, BA, BA, 5, 3, 23, 3, 3, 25, 2, 2, 25, 3, 3, 23, 3, 5, BA, BA, BA, BA])  # 14####         |
    core.add_column(materials=[BA, BA, 5, 5, 3, 3, 3, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 5, 5, BA, BA])  # 17####           |
    core.add_column(materials=[BA, BA, 5, 3, 3, 3, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 3, 3, 3, 5, BA, BA])  # 18####         |
    core.add_column(materials=[BA, 5, 3, 3, 25, 2, 1, 21, 2, 1, 23, 1, 2, 21, 1, 2, 25, 3, 3, 5, BA])  # 19####           |
    core.add_column(materials=[BA, 5, 3, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 3, 5, BA])  # 20####         |
    core.add_column(materials=[5, 3, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 3, 5])  # 21####           |
    core.add_column(materials=[BA, 5, 3, 25, 1, 1, 23, 1, 2, 23, 1, 1, 23, 2, 1, 23, 1, 1, 25, 3, 5, BA])  # 20####         |
    core.add_column(materials=[5, 3, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 3, 5])  # 21####           v
    core.add_column(materials=[5, 3, 3, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 3, 5])  # 22####
    core.add_column(materials=[5, 23, 3, 2, 26, 1, 1, 23, 2, 1, 21, 1, 2, 23, 1, 1, 21, 2, 3, 23, 5])  # 21----MITAD DEL NUCLEO
    core.add_column(materials=[5, 3, 3, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 3, 5])  # 22####
    core.add_column(materials=[5, 3, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 3, 5])  # 21####
    core.add_column(materials=[BA, 5, 3, 25, 1, 1, 23, 1, 2, 23, 1, 1, 23, 2, 1, 23, 1, 1, 25, 3, 5, BA])  # 20####
    core.add_column(materials=[5, 3, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 3, 3, 5])  # 21####
    core.add_column(materials=[BA, 5, 3, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 3, 5, BA])  # 20####
    core.add_column(materials=[BA, 5, 3, 3, 25, 2, 1, 21, 2, 1, 23, 1, 2, 21, 1, 2, 25, 3, 3, 5, BA])  # 19####
    core.add_column(materials=[BA, BA, 5, 3, 3, 3, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 3, 3, 3, 5, BA, BA])  # 18####
    core.add_column(materials=[BA, BA, 5, 5, 3, 3, 3, 1, 2, 1, 2, 1, 2, 1, 3, 3, 3, 5, 5, BA, BA])  # 17####
    core.add_column(materials=[BA, BA, BA, BA, 5, 3, 23, 3, 3, 25, 2, 2, 25, 3, 3, 23, 3, 5, BA, BA, BA, BA])  # 14####
    core.add_column(materials=[BA, BA, BA, BA, 5, 5, 3, 3, 3, 3, 2, 3, 3, 3, 3, 5, 5, BA, BA, BA, BA])  # 13####
    core.add_column(materials=[BA, BA, BA, BA, BA, BA, 5, 5, 5, 3, 3, 3, 3, 5, 5, 5, BA, BA, BA, BA, BA, BA])  # 10####
    core.add_column(materials=[BA, BA, BA, BA, BA, BA, BA, BA, 5, 5, 5, 5, 5, BA, BA, BA, BA, BA, BA, BA, BA])  # 5 ####
    with open('aer.out.sc5', 'w') as fod: fod.write(core.string)
    return

def test_sc5():
    aer = AerModel()
    aer.to_file_sc5('aer.out.sc5')
    return


if __name__ == '__main__':
    test_sc5()
    pass


