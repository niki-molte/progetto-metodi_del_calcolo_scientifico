from dataclasses import dataclass, field


@dataclass
class Results:
    _nit: int = field(init=False, repr=False)
    _err: float = field(init=False, repr=False)
    _tol: float = field(init=False, repr=False)
    _tim: float = field(init=False, repr=False)

    def __init__(self, nit, err, tol, tim):
        self._nit = nit
        self._err = err
        self._tol = tol
        self._tim = tim

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
        if not isinstance(value, float) and value < 0:
            raise ValueError("value should be a positive float")

    @property
    def tim(self) -> float:
        return self._tim

    @tim.setter
    def tim(self, value: float):
        if not isinstance(value, float) and value < 0:
            raise ValueError("value should be a positive float.")
        self._tim = float(value)

    def __str__(self) -> str:
        return f"Results(number of iteration={self.nit}, \n " \
               f"       error={self.err:.15e}, \n" \
               f"        tolerance={self.tol:.15e}, \n" \
               f"        time={self.tim:.15e} seconds \n" \
               f")"
