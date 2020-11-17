

import math
import re
from collections import Counter



class CosineSimilarity:
    def __init__(self):
        print()

    def calculateCosine(self, T1, T2):
        common = set(T1.keys()) & set(T2.keys())
        temp = sum([T1[x] * T2[x] for x in common])
        K1 = sum([T1[x] ** 2 for x in list(T1.keys())])
        K2 = sum([T2[x] ** 2 for x in list(T2.keys())])
        temp1 = math.sqrt(K1) * math.sqrt(K2)
        if not temp1:
            return 0.0
        else:
            return float(temp) / temp1


    def text2Vec(self, text):
        words = re.compile(r"\w+").findall(text)
        return Counter(words)

    def getCosineSimilarityBetweenSentences(self,text1,text2):
        vector1 = self.text2Vec(text1)
        vector2 = self.text2Vec(text2)

        cosine = self.calculateCosine(vector1, vector2)
        print("Cosine:", cosine)
        return cosine

x=CosineSimilarity()
print(x.getCosineSimilarityBetweenSentences("Comment ça va ?","Coment ça  ?"))
