import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Base:
    
    from .Algorithm_3.ANN.index import ANN

    def __init__(self, data):
        self.data = pd.read_csv(data)