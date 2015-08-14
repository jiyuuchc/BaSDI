from typing import Callable, Sequence, Tuple

from jax import Array
from jax.typing import ArrayLike

SMCHistory = Sequence[Tuple[ArrayLike, ArrayLike]]
RNG = ArrayLike
TransModel = Callable