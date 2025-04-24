from dataclasses import dataclass, field


@dataclass
class Results:
    _nit: int = field(init=False, repr=False)
    _err: float = field(init=False, repr=False)
    _tol: float = field(init=False, repr=False)
    _tim: float = field(init=False, repr=False)
    _dim: int = field(init=False, repr=False)
    _mem: float = field(init=False, repr=False)
    _mep: float = field(init=False, repr=False)

    def __init__(self, nit, err, tol, tim, dim, mem, mep):
        self.nit = nit
        self.err = err
        self.tol = tol
        self.tim = tim
        self.dim = dim
        self.mem = mem / 1024
        self.mep = mep / 1024

    @property
    def nit(self) -> int:
        return self._nit

    @nit.setter
    def nit(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("value should be a positive int")
        self._nit = value

    @property
    def err(self) -> float:
        return self._err

    @err.setter
    def err(self, value: float):
        if not isinstance(value, float) or value < 0:
            raise ValueError("value should be a positive float.")
        self._err = float(value)

    @property
    def tol(self) -> float:
        return self._tol

    @tol.setter
    def tol(self, value: float):
        if not isinstance(value, float) or value < 0:
            raise ValueError("value should be a positive float")
        self._tol = value

    @property
    def tim(self) -> float:
        return self._tim

    @tim.setter
    def tim(self, value: float):
        if not isinstance(value, float) or value < 0:
            raise ValueError("value should be a positive float.")
        self._tim = value

    @property
    def dim(self) -> float:
        return self._dim

    @dim.setter
    def dim(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("value should be a positive integer.")
        self._dim = value

    @property
    def mem(self) -> float:
        return self._mem

    @mem.setter
    def mem(self, value: float):
        if not isinstance(value, float) or value < 0:
            raise ValueError("value should be a positive float.")
        self._mem = value

    @property
    def mep(self) -> float:
        return self._mep

    @mep.setter
    def mep(self, value: float):
        if not isinstance(value, float) and value < 0:
            raise ValueError("value should be a positive float.")
        self._mep = value

    def __str__(self) -> str:
        return (f"Results(\n "
                f"       number of iteration={self.nit}, \n "
                f"       error={self.err:.17e}, \n"
                f"        tolerance={self.tol:.0e}, \n"
                f"        matrix dim={self.dim:.0e}, \n"
                f"        time={self.tim:.17e} seconds, \n"
                f"        memory usage={self.mem:.17f} KiloBytes, \n"
                f"        memory peak={self.mep:.17f} KiloBytes. \n"
                f")")
