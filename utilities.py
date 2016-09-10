import numpy as np
alpha_weights={'a':1,'e':1,'i':1,'o':1,'u':1,'x':4,'z':4,'q':5}
consonants = set("bcdfghjklmnpqrstvwxyz")


class wordweight:
    def __init__(self):
        self.wordWeight=0
    def wdweight(self,s):
        for j in s:
            try:
                self.wordWeight+=alpha_weights[j]
            except:
                self.wordWeight+=1.5
        # try:
        #     for j in range(len(s)-2):
        #         if (s[j] in consonants and s[j+1] in consonants and s[j+2] in consonants):
        #             self.wordWeight=self.wordWeight*1.5
        # except:
        #     self.wordWeight=self.wordWeight
        return self.wordWeight

class vowelCount:

    def __init__(self):
        self.count_vowels=0

    def vCount(self,s):
        for j in s:
            if j in['a','e','i','o','u']:
                self.count_vowels+=1
        return self.count_vowels


