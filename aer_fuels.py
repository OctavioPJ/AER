from itertools import groupby

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:04:27 2019

@author: CNEA
"""


class MATERIAL(object):
    _KEY__ = ''
    _NUM__ = None
    _TYPE__ = 'MATERIAL'

    def __init__(self, *args, **kwargs):
        self._NUM__ = args
        self._KEY__ = kwargs['KEY']

    def string(self, *args, **kwargs):
        return '^^ ' + self._TYPE__ + ' = ' + \
               '(' * (len(self._NUM__) != 1) + \
               ' '.join(map(str, self._NUM__)) + \
               ')' * (len(self._NUM__) != 1) + \
               ('   KEY= ' + self._KEY__) * (
                   1 if not hasattr(self, '_ax') else 0)  # TODO esta linea está recontra improlija


class FUEL_ELEMENT(MATERIAL):
    _AX_ELEM = 1

    def __init__(self, *args, **kwargs):
        super(FUEL_ELEMENT, self).__init__(*args, **kwargs)
        try:
            Ax_elements = iter(kwargs['AXIAL'])
            self._AX_ELEM = sum(map(lambda ax_slice: ax_slice[1].stop - ax_slice[1].start, Ax_elements))
            self._ax = kwargs['AXIAL']
        except TypeError as te:
            self._AX_ELEM = kwargs['AXIAL']
            self._ax = [self._KEY__ for _ in range(self._AX_ELEM)]
        self._TYPE__ = 'FUEL ELEMENT'

    @property
    def ax_elem(self):
        return self._AX_ELEM

    def __setitem__(self, ax_slice, key):
        assert isinstance(key, str), 'Tipo de variable ' + str(type(key))+' cuando se esperaba str'
        if hasattr(ax_slice, 'indices'):
            for k in range(*ax_slice.indices(self._AX_ELEM)):
                self._ax[k] = key
        else:
            self._ax[ax_slice] = key

    def string(self, *args, **kwargs):
        _ax = []
        for group, items in groupby(enumerate(self._ax), lambda a: a[1]):
            _group = list(items)
            start = _group[0][0]
            stop = _group[-1][0]
            _ax.append((group, slice(start, stop)))

        return super(FUEL_ELEMENT, self).string(*args, **kwargs) + (
                ' *\n' + ' ' * len(super(FUEL_ELEMENT, self).string(*args, **kwargs))).join(
            map(lambda ax_slice:
                ' KEY= ' + ax_slice[0] + ' AXIAL ELEMENT={0:2d} TO {1:2d}'.format(
                    ax_slice[1].start + 1, ax_slice[1].stop + 1),
                _ax))

    pass  # FUEL_ELEMENT


class CONTROL_ROD(FUEL_ELEMENT):
    def __init__(self, *args, **kwargs):
        assert ('IN' in kwargs)
        Kwargs = kwargs.copy()
        Kwargs.update({'KEY': kwargs['KEY'] + '/IN=' + kwargs['IN']})
        super(CONTROL_ROD, self).__init__(*args, **Kwargs)
        self._TYPE__ = 'CONTROL ROD'

    pass  # CONTROL_ROD


class SECCION8(object):
    def __init__(self, library):
        self.__library = library
        self.__materials = []

    def AddMaterial(self, material):
        assert isinstance(material, MATERIAL)
        self.__materials.append(material)

    def to_file(self, file):
        with open(file, 'w') as fod:
            fod.write('^^ LIBRARY = ' + self.__library + '\n')
            fod.write('^^ SEC 8 GROUP from library\n')
            AX_ELEM = 0
            for material in self.__materials:
                if hasattr(material, 'ax_elem') and material.ax_elem > AX_ELEM:
                    AX_ELEM = material.ax_elem
                fod.write(material.string() + '\n')
            #                print(AX_ELEM)
            #                print(material._get_ax_elem())
            fod.write('^^ INSERTION MAPPING 0 {}\n'.format(AX_ELEM))
            fod.write('^^ FISSION SPEC FROM LIBRARY\n')
        return

    def ChangeMaterial(self, NewKeyword, Start, Stop, *MatIndex):

        Start = int(Start)
        Stop = int(Stop)

        if Start != Stop:
            for Indx in MatIndex:
                self.__materials[Indx][Start:Stop] = NewKeyword
        else:
            for Indx in MatIndex:
                self.__materials[Indx][Start] = NewKeyword
        return

    pass  # SECCION8


def main_test():
    print(MATERIAL(1, KEY='AGUA').string())
    print(FUEL_ELEMENT(2, KEY='TIPO5', AXIAL=10).string())
    print(CONTROL_ROD(4, 3, 4, 5, 6, KEY='TIPO9', AXIAL=9, IN='TIPO3').string())
    return


def main():
    aer_sc8 = SECCION8('aer')
    aer_sc8.AddMaterial(MATERIAL(1, KEY='AGUA'))
    alf = FUEL_ELEMENT(2, KEY='TIPO5', AXIAL=10)
    alf[1:5] = 'TIPO6'
    aer_sc8.AddMaterial(alf)
    aer_sc8.AddMaterial(CONTROL_ROD(4, 3, 4, 5, 6, KEY='TIPO9', AXIAL=9, IN='TIPO3'))
    aer_sc8.to_file('aer_output_python.mat')
    return


class C:
    def __init__(self, **kwargs):
        self._val = 1
        self.__b = {0: 1}
        # self.axial = kwargs['AXIAL']

    @property
    def val(self):
        return self._val

    @property
    def b(self):
        return self.__b

    def __getitem__(self, items):
        print(items)
        return items

    def __setitem__(self, key, value):
        self[key] = value

    def show(self, *args):
        print(args)
        return


if __name__ == '__main__':
    # aer_sc8 = SECCION8('aer')
    # aer_sc8.AddMaterial(MATERIAL(1, KEY='AGUA'))
    alf = FUEL_ELEMENT(2, KEY='TIPO5', AXIAL=10)
    # alf[1:5] = 'TIPO6'
    # aer_sc8.AddMaterial(alf)
    # aer_sc8.AddMaterial(CONTROL_ROD(4, 3, 4, 5, 6, KEY='TIPO9', AXIAL=9, IN='TIPO3'))
    # aer_sc8.to_file('aer_output_python.mat')
    # c = C()
    # c.show(*map(lambda CR: '{}b'.format(CR), [21, 23]))
