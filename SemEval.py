import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk.corpus.reader.wordnet
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from nltk import FreqDist
from nltk.corpus import brown



import re
import nltk
stemmer=SnowballStemmer('english')
data=open("C:\Users\Krishna\Desktop\SemEval\cwi\cwi_training_allannotations.txt",'r')
dataval=[]
vectorizer = TfidfVectorizer()
transformer = TfidfTransformer()

for i in range (2230):
    dataval= [np.append((dataval),[data.readline()])]
dataval=np.array(dataval)
print dataval[0][2000],len(dataval[0][:])
data2=[[dataval[0][i]] for i  in range (2230)]
data2=[re.split(" [.]['\t']",dataval[0][i],maxsplit=1)for i in range((2230))]
#print (data2)
features=[[]for i in range(len(data2))]#for j in range (len(data2))]
word=[[]for i in range(len(data2))]
sentence=[[]for i in range(len(data2))]
print  "kakka"
for i in range (len(data2)):
    k=0
    features[i]=[j for j in data2[i]]

    for j in features[i]:
        k=k+1
        if (k==2):
            word[i]= j
        else:
            sentence[i] = j
print (word[10])

for i in range(len(word)):
    if(word[i]==[]):
        sentence[i]=[]


word=filter(None,word)
sentence=filter(None,sentence)
word=[re.split('[\t|\n]',word[i],maxsplit=25)for i in range(len(word))]
complexWords=np.zeros(len(word))
for i in range(len(word)):
    currentWord=word[i]

    for j in range(len(currentWord)-21,len(currentWord)):
        print currentWord[j]
        if(currentWord[j]=='1'):
            complexWords[i]=1
            break

complexWords=complexWords.reshape(-1,1)
print complexWords.shape
print complexWords

wordInSent = []
k = 0

wordInSent=[nltk.tokenize.word_tokenize(sentence[i]) for i in range(len(sentence))]



#wordInSent=[[stemmer.stem(w) for w in wordInSent[i]] for i in range (len(wordInSent))]
"""for i in range (len(wordInSent)):
    wordInSent[i]=[j.lower() for  j in wordInSent[i]]
vect_representation=vectorizer.fit()
tfidf=transformer.fit_transform(vect_representation)

#print transformer.idf_



vect_representation= map(vectorizer.fit_transform,wordInSent)
vecx=np.array(vect_representation)
print
print vecx.shape
"""
fdist=FreqDist(brown.words())
word_count=[]
for i in range(50):
    m=wordInSent[i]
    l=[]
    for j in m:
        #if(j==)
        word_count.append(fdist.freq(j))
    #word_count.append(l)
print word_count[2]
for i in range (len(word_count)):
    word_count[i]=[np.array(word_count[i])]
word_count=np.array(word_count)
print word_count
"""
classifier=DecisionTreeClassifier(criterion='gini')
classify=classifier.fit((word_count[0:2000]),complexWords[0:2000])
ypred=classifier.predict((word_count[2001:]))
target_names=['0','1']
print(classification_report(complexWords[2001:], ypred, target_names=target_names))
"""