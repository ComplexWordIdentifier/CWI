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
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from nltk import FreqDist
from nltk.corpus import brown
import re
from synonymcount import synonymcount
from utilities import vowelCount,wordweight
synobj=synonymcount()
vc=vowelCount()
ww=wordweight()

import nltk
size=5000
# class datacleaning:

def readfile(filename):
    data=open(filename,'r')
    dataval=[]
    for i in range ((size)):
        dataval= [np.append((dataval),[data.readline()])]
    dataval=np.array(dataval)
    data2=[re.split(" [.!?]['\t']",dataval[0][i],maxsplit=1)for i in range((5000))]
    return data2

# getting LHS and RHS data
def prepare_from_all_annotated(data2):
    data21=[]
    for i in range(len(data2)):
        for j in range(len(data2[i])):
            if j==0:
                data21.append(data2[i][j])
    # print data21
    data22=[]
    for i in range(len(data2)):
        for j in range(len(data2[i])):
            if j==1:
                data22.append(data2[i][j])
    return data21,data22


def obtain_x_and_y(data22,data21):
    rhs=[]
    for i in range(len(data22)):
        rhs.append([re.split("[\t]",data22[i],maxsplit=21)])
    # weights=[]
    complexity=dict()
    # for i in range(len(rhs)):
    #     weights.append([])
    for i in range(len(rhs)):
        for k in rhs[i]:
            count=0
            # print k
            for j in range(len(k)):
                if j>1:
                   if(k[j] =='1' or k[j] =='1\n'):
                       count=1
        complexity.update({k[0].lower():count})
    # print complexity['compounds']
    data21=set(data21)
    data21=list(data21)
    # print data21

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
    y=np.zeros(len(allwords))
    # print allwords
    for i in range(len(allwords)):
        try:
            y[i]=int((complexity[allwords[i]]))

        except :
            y[i]=0
    # print y
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
    return x,y
def any_classification(classifier,x,y):
    classify=classifier.fit((x[0:int(len(x)*.5)]),y[0:int (len(y)*.5)])
    ypred=classifier.predict(x[int(len(x)*.5)+1:])
    # target_names=['1','2','3','4','5']
    target_names=['0','1']
    print(classification_report(y[int (len(y)*.5)+1:], ypred, target_names=target_names))



