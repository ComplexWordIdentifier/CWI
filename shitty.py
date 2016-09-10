# from nltk.corpus import brown
# from nltk import FreqDist
# from nltk.corpus import wordnet as wn
# import nltk.corpus.reader.wordnet
# from synonymcount import synonymcount
#
# # fdist=FreqDist(brown.words())
# s="unwittingly"
# for synset in wn.synsets(s):
#     for lemma in synset.lemmas():
#         print lemma.name()
# r=synonymcount()
# print r.synCount(s)
import modular
data=[]
data=modular.readfile('C:\Users\Krishna\Desktop\SemEval\cwi\cwi_testing_annotated.txt')
data1=[]
data2=[]
data1,data2=modular.prepare_from_all_annotated(data)
x=[]
y=[]
x,y=modular.obtain_x_and_y(data2,data1)
print y



