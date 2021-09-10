import toad
import numpy as np
from numpy import array
from numpy import int64

class LR_preprocessing:
    def __init__(self, config):
        self.keep_cols = config["keep_cols"]
        self.bins = config["bins"]
        self.woes = config["woes"]
        self.binner = toad.transform.Combiner()
        self.binner.set_rules(self.bins)
        self.woer = toad.transform.WOETransformer()
        self.woer.rules = self.woes

    @classmethod
    def from_configfile(cls, file_path, encoding="utf-8"):
        from numpy import array
        from numpy import int64
        with open(file_path, "r", encoding=encoding) as f:
            config = eval(f.read())
        return cls(config)

    def transform(self, X):
        X = X[self.keep_cols]
        X_binned = self.binner.transform(X)
        X_woed = self.woer.transform(X_binned)
        return X_woed

