# -*- coding: utf-8 -*-
"""
@author: Steffany
"""
from nltk import pos_tag, bigrams, ngrams
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

#create corpus and tokenize them
corpus13 = PlaintextCorpusReader('2013/', '.*',encoding='latin-1')
corpus14 = PlaintextCorpusReader('2014/', '.*',encoding='latin-1')
lower = stopwords.words('english')
titled = [x.title() for x in stopwords.words('english')]
stop  = set(lower + titled)

all13 = corpus13.words()
fc13 = []

all14 = corpus14.words()
fc14 = []

def no_stopwords(old, new):
    #removes stopwords from corpus + creates new corpus without stop words
    for w in old:
        if w not in stop:
            new.append(w)
            
no_stopwords(all13,fc13)
no_stopwords(all14,fc14)
del lower
del titled

#pos tag corpus w/o stopwords
fc13pos = pos_tag(fc13)
fc14pos = pos_tag(fc14)

#bigram pos tagged corpus
bi13pos = list(bigrams(fc13pos))
bi14pos = list(bigrams(fc14pos))
bipos = bi13pos + bi14pos

#trigram pos tagged corpus
tri13pos = list(ngrams(fc13pos,3))
tri14pos = list(ngrams(fc13pos,3))
tripos = tri13pos + tri14pos

#import training datasets
ceo = PlaintextCorpusReader('all',"ceo.txt",encoding='latin-1')
ceo_pos = pos_tag(ceo.words())
biceo = list(bigrams(ceo_pos))

percent = PlaintextCorpusReader('all', "percentage.txt",encoding='latin-1')
percent_pos = pos_tag(percent.words())
biper = list(bigrams(percent_pos))
triper = list(ngrams(percent_pos,3))

company = PlaintextCorpusReader('all', "companies.txt",encoding='latin-1')
company_pos = pos_tag(company.words())
bicomp = list(bigrams(company_pos))
tricomp = list(ngrams(company_pos,3))


""" 
Models are all written below
The general format of a model is as follows:
    feature extraction
    creating and labeling a training set
    training a Naive Bayes Classifier 
    extracting desired type to list using the classifer
    writing list to file
"""


#CEO MODEL
#feature extraction
def ceo_features(name):
    features = {}
    features["firstname_pos"] = name[0][1] #pos of 1st name
    features["lastname_pos"] = name[1][1] #pos of 2nd name
    return features

#labeling training set
labeled_ceos = ([(ceo_name,'ceo') for ceo_name in biceo] +
               [(ceo_name,'not ceo') for ceo_name in biper])
ceo_fset = [(ceo_features(n),ceo) for (n,ceo) in labeled_ceos] #feature set
ceo_classifier = NaiveBayesClassifier.train(ceo_fset) #training classfier
ceo_classifier.classify(ceo_features(bipos[1])) #test
ceo_classifier.show_most_informative_features(5)

#extracting ceos
ceo_list = []
for i in range(len(bipos)):
    if ceo_classifier.classify(ceo_features(bipos[i])) == 'ceo':
        ceo_list.append(bipos[i])
tpceo = len(ceo_list)/len(bipos)

#writing to file
ceo_names = open('ceo_names.txt','w')
for item in ceo_list:
  ceo_names.write(str(item))
ceo_names.close()


#PERCENT MODEL
def perc_features(per):
    features = {}
    features["number"] = per[0][1] #pos of number
    features["sign"] = per[1][1] #pos of sign
    return features

labeled_percent = ([(perc,'percent') for perc in biper] +
               [(perc,'not percent') for perc in bicomp])
perc_fset = [(perc_features(n),perc) for (n,perc) in labeled_percent]
perc_classifier = NaiveBayesClassifier.train(perc_fset)
perc_classifier.classify(perc_features(bipos[1]))
perc_classifier.show_most_informative_features(5)

perc_list = []
for i in range(len(bipos)):
    if perc_classifier.classify(perc_features(bipos[i])) == 'percent':
        perc_list.append(bipos[i])
tpperc = len(perc_list)/len(bipos)

per_extr = open('extracted_percents.txt','w')
for item in perc_list:
  per_extr.write(str(item))
per_extr.close()


#COMPANY MODEL
def comp_features(comp):
    features = {}
    features["first_word"] = comp[0][1] #pos of 1st word
    features["second_word"] = comp[1][1] #pos of 2nd word
    features["third_word"] = comp[2][1] #pos of 2nd word
    return features

labeled_comps = ([(comp,'company') for comp in tricomp] +
               [(comp,'not company') for comp in triper])
comp_fset = [(comp_features(n),comp) for (n,comp) in labeled_comps]
comp_classifier = NaiveBayesClassifier.train(comp_fset)
comp_classifier.classify(comp_features(tripos[1]))
comp_classifier.show_most_informative_features(5)

comp_list = []
for i in range(len(tripos)):
    if comp_classifier.classify(comp_features(tripos[i])) == 'company':
        comp_list.append(tripos[i])
tpcomp = len(comp_list)/len(tripos)

comp_names = open('company_names.txt','w')
for item in comp_list:
  comp_names.write(str(item))
comp_names.close()