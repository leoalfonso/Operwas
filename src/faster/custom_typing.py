from typing import Callable

import numpy as np
import numpy.typing as npt

Connection = tuple[int, int, float]
Coordinate = tuple[float, float]
DataInterpolator = Callable[[float], float]

Numpy1dFloat = npt.NDArray[np.float_]
Numpy1dString = npt.NDArray[np.str_]
Numpy1dInt = npt.NDArray[np.int_]

Scalar = int | float | str
