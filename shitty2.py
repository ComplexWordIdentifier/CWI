import modular
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk.corpus.reader.wordnet
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk import regexp_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from nltk import FreqDist
from nltk.corpus import brown
from synonymcount import synonymcount
from utilities import vowelCount,wordweight
import re
import nltk
PATH_NAME='C:\Users\Krishna\Desktop\SemEval\\abcd.txt'
data =modular.readfile(PATH_NAME)
synobj=synonymcount()

vc=vowelCount()
ww=wordweight()


data21=[]
for i in range(len(data)):
    for j in range(len(data[i])):
        if j==0:
            data21.append(data[i][j])
data21=set(data21)
data21=list(data21)
data211=[]
"""HERE IS WHERE U CHANGE THE RANGE to range(len(data21))"""
for i in range(len(data21)):
    # print data21[i]
    text=data21[i].split()
    data211.append(text)
# print (data211)
stop=set(stopwords.words('english'))
stop =list(stop)
stop=[unicode(stop[i]) for i in range(len(stop))]
allwords=[]
for i in data211:
    for j in i:
        if j not in stop :
            allwords.append(j.lower())
# print allwords
for i in range(len(allwords)):
    allwords=[re.sub(r'[^\w\s]','',s) for s in allwords]
allwords=set(allwords)
allwords=list(allwords)
fdist=FreqDist(brown.words())

x=[]
for i in range(len(allwords)):
    x.append([])
for i in range(len(allwords)):
    x[i].append(fdist.freq(allwords[i]))
    x[i].append(len(allwords[i]))
    x[i].append(synobj.synCount(allwords[i]))
    x[i].append(ww.wdweight(allwords[i]))
    x[i].append(vc.vCount(allwords[i]))
    x[i].append(synobj.len_of_synonyms(allwords[i]))
XTest=x
TestAllWords=allwords