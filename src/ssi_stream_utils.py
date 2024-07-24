import xml.etree.ElementTree as ET
import numpy as np
from enum import Enum


class FileTypes(Enum):
        UNDEF = 0
        BINARY = 1
        ASCII = 2
        BIN_LZ4 = 3

class NPDataTypes(Enum):
    UNDEF = 0
    #CHAR = 1
    #UCHAR = 2
    SHORT = np.int16
    USHORT = np.uint16
    INT = np.int32
    UINT = np.uint32
    LONG = np.int64
    ULONG = np.uint64
    FLOAT = np.float32
    DOUBLE = np.float64
    LDOUBLE = np.float64
    #STRUCT = 12
    #IMAGE = 13
    BOOL = np.bool_

def string_to_enum(enum, string):
    for e in enum:
        if e.name == string:
            return e
    raise ValueError('{} not part of enumeration  {}'.format(string, enum))



class Stream:
    def __init__(self, path=None):
        self.ftype = string_to_enum(FileTypes, "UNDEF")
        self.sr = 0
        self.dim = 0
        self.byte = 4
        self.type = "UNDEF"
        self.delim = ""
        self.chunks = []
        self.data = None

        if path:
            self.load(path)

    def load_header(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        chunks = 0

        for child in root:
            if child.tag ==  'info':
                for key,val in child.attrib.items():
                    if key == 'ftype':
                        self.ftype = string_to_enum(FileTypes, val)
                    elif key == 'sr':
                        self.sr = float(val)
                    elif key == 'dim':
                        self.dim = int(val)
                    elif key == 'byte':
                        self.byte = int(val)
                    elif key == 'type':
                        self.type = string_to_enum(NPDataTypes, val).value
                    elif key == 'delim':
                        self.delim = val
            elif child.tag == 'chunk':
                f, t, b, n = 0, 0, 0, 0
                for key,val in child.attrib.items():
                    if key == 'from':
                        f = float(val)
                    elif key == 'to':
                        t = float(val)
                    elif key == 'num':
                        n = int(val)
                    elif key == 'byte':
                        b = int(val)
                chunks += 1
                self.chunks.append( [f, t, b, n] )
        return self

    def load_data(self, path):
        if self.ftype == FileTypes.ASCII:
            self.data = np.loadtxt(path, dtype=self.type, delimiter=self.delim)
        elif self.ftype == FileTypes.BINARY:
            num = np.sum(self.chunks, axis=0, dtype=np.int32)[3]
            self.data = np.fromfile(path, dtype=self.type).reshape(num, self.dim)
        else:
            raise ValueError('FileType {} not supported'.format(self))
        return self.data

    def load(self, path):
        self.load_header(path)
        self.load_data(path + '~')
        return self
