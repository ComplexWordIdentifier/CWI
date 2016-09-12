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
from synonymcount import synonymcount
synobj=synonymcount()
import re
import nltk
import matplotlib.pyplot as pl

dataval=[]
data=open("C:\Users\Krishna\Desktop\SemEval\cwi\cwi_testing_annotated.txt",'r')
for i in range ((1000)):
    dataval= [np.append((dataval),[data.readline()])]
dataval=np.array(dataval)

data2=[re.split(" [.!?]['\t']",dataval[0][i],maxsplit=1)for i in range((1000))]
# print data2
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
# print data22
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
                   count+=1
    complexity.update({k[0].lower():count})
# print complexity['compounds']

pat = r'''\ '''
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
freqComplex=[]
freqSimple=[]
x=[]
for i in range(len(allwords)):
    x.append([])
for i in range(len(allwords)):
    x[i].append(fdist.freq(allwords[i]))
    if (y[i]==1):
        freqComplex.append(fdist.freq(allwords[i]))
    else:
        freqSimple.append(fdist.freq(allwords[i]))
    x[i].append(len(allwords[i]))
    x[i].append(synobj.synCount(allwords[i]))
    x[i].append(synobj.len_of_synonyms(allwords[i]))



"""NO OF VOWELS"""
complex_vowels=[]
word_weights_complex=[]
simple_vowels=[]
word_weights_simple=[]
"""all vowels 1,consonants 1.5,x&z 4,q 5"""
alpha_weights={'a':1,'e':1,'i':1,'o':1,'u':1,'x':4,'z':4,'q':5}
consonants = set("bcdfghjklmnpqrstvwxyz")

for i in range(len(allwords)):
    wordWeight=0
    for j in allwords[i]:
        try:
            wordWeight+=alpha_weights[j]
        except:
            wordWeight+=1.5
    try:
        for j in range(len(allwords[i])-2):
            if (allwords[i][j+2] in consonants and allwords[i][j+1] in consonants and allwords[i][j] in consonants ):
                wordWeight=wordWeight*1.5
    except:
        wordWeight=wordWeight

    if y[i]==1:
        word_weights_complex.append(wordWeight)
    else:
        word_weights_simple.append(wordWeight)
word_weights_simple=np.array(word_weights_simple)
word_weights_complex=np.array(word_weights_complex)
print word_weights_complex,word_weights_complex.mean(),word_weights_complex.std()
print word_weights_simple,word_weights_simple.mean(),word_weights_simple.std()


for i in range(len(allwords)):
    count_vowels=0
    for j in allwords[i]:
        if j in['a','e','i','o','u']:
            count_vowels+=1
    if(y[i]==1):
        try:
            complex_vowels.append(count_vowels)
        except:
            complex_vowels.append(0.0)
    else:
        try:
            simple_vowels.append(count_vowels)
        except:
            simple_vowels.append(0)
    print allwords[i],count_vowels
complex_vowels=np.array(complex_vowels)
simple_vowels=np.array(simple_vowels)

print complex_vowels,'mean:',complex_vowels.mean(),'std:',complex_vowels.std()
fig, axes = pl.subplots(nrows=1, ncols=2)
axes[0, 0].boxplot(freqSimple, labels=['brown freq simple words '],showmeans=True)
axes[0, 1].boxplot(freqComplex[0:250], labels=['brown freq complex words'],showmeans=True)
# axes[1, 0].boxplot(word_weights_simple, labels=['word weight simple'],showmeans=True)
# axes[1, 1].boxplot(word_weights_complex, labels=['word weight complex'],showmeans=True)

pl.show()

# print simple_vowels,'mean:',simple_vowels.mean(),'std:',simple_vowels.std()