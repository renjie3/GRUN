import pyvene_custom as pv
import json


class ReftConfig(pv.IntervenableConfig):
    """
    Reft config for Reft methods.
    """
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)