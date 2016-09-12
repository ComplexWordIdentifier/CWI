from nltk.corpus import wordnet as wn

class synonymcount:

    def __init__(self):
        self.no_of_syns=1
        self.synLength=0
        self.lemmaLength=0
        self.synName=[]
    # def synValues(self,s):
    #     for synset in wn.synsets(s):
    #         for lemma in synset.lemmas():
    #             self.synName.append(lemma.name())
    #     return self.synName[0]

    def synCount(self,s):
        for synset in wn.synsets(s):
            for lemma in synset.lemmas():
                self.no_of_syns+=1
        return self.no_of_syns
    def len_of_synonyms(self,s):
        for synset in wn.synsets(s):
            for lemma in synset.lemmas():
                self.synLength+=len(lemma.name())
                self.no_of_syns+=1
        return (self.synLength/self.no_of_syns)



