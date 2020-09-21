import numpy as np
from typing import NamedTuple, List


class PARAMConfig(NamedTuple):
    N_EPISODE: int  # total child networks to be created
    NUM_EPOCHS: int
    CHILD_BATCHSIZE: int
    HIDDEN_SIZE: int
    BETA: float
    INPUT_SIZE: int
    ACTION_SPACE: int
    NUM_STEPS: int
    GAMMA: float
    LEARNING_RATE: float


class NetworkConfig(NamedTuple):
    INPUT_SIZE: int
    HIDDEN_SIZE: int
    NUM_STEPS: int
    ACTION_SPACE: int
    LEARNING_RATE: float
