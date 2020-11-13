import os
import numpy as np
import pandas as pd
import os
import pickle
class Preprocessing:
    def preProcessData(self):
        freList,engList = [],[]
        dataPath = "Data/eng-french.txt"
        with open(dataPath, "r", encoding="utf-8") as pointer:
            each = pointer.read().split("\n")
        for line in each[:10000]:
            english, french, extras = line.split("\t")
            engList.append(english)
            freList.append(french)
        return engList,freList

