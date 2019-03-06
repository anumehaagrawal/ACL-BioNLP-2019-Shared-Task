import pandas as pd 
import xml.etree.ElementTree as ET
import nltk
import re, math
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import gensim.downloader as api
import spacy
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordNERTagger
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

WORD = re.compile(r'\w+')
nlp = spacy.load('en_core_web_sm')


def cleaned_words(myString):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    stemmer = PorterStemmer()
    result = [' '.join([stemmer.stem(w).lower() for w in x.split()]) for x in tokens if x.lower() not in stopwords.words('english')]
    return result
    
def stem_sentence(sentence):
    ps = PorterStemmer()
    return ps.stem(sentence)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return len(lst3) 

def get_jaccard_sim(a, b): 
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def verbs(sentence1,sentence2):
    count=0
    text1 = nltk.word_tokenize(sentence1)
    result1 = nltk.pos_tag(text1)
    text2=nltk.word_tokenize(sentence2)
    result2=nltk.pos_tag(text2)
    for i in result1:
        for j in result2:
            if i[1]=='VB' and j[1]=='VB':
                if i[0]==j[0]:
                    count+=1
    return count

def nouns(sentence1,sentence2):
    count=0
    text1 = nltk.word_tokenize(sentence1)
    result1 = nltk.pos_tag(text1)
    text2=nltk.word_tokenize(sentence2)
    result2=nltk.pos_tag(text2)
    for i in result1:
        for j in result2:
            if i[1]=='NN' and j[1]=='NN':
                if i[0]==j[0]:
                    count+=1
    return count

def levenshtein(sentence1, sentence2):  
    size_x = len(sentence1) + 1
    size_y = len(sentence2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if sentence1[x-1] == sentence2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def training_set(tree,x_values,y_values):
    
    root = tree.getroot()
    count=0
    for pair in root.findall('pair'):
        chq = pair.find('chq').text
        faq = pair.find('faq').text
        value = pair.get('value')
        if(value=="true"):
                y_values.append(1)
        else:
                y_values.append(0)
        tokens_chq = cleaned_words(chq)
        tokens_faq = cleaned_words(faq)
        sent_chq=stem_sentence(chq)
        sent_faq=stem_sentence(faq)
        #feature1: word overlap
        word_overlap = intersection(tokens_chq,tokens_faq)/len(tokens_chq)
        #feature2: common bigrams
        bigram_chq = list(ngrams(tokens_chq, 2)) 
        bigram_faq = list(ngrams(tokens_faq, 2)) 
        bigram_overlap = intersection(bigram_chq,bigram_faq)/len(bigram_chq)
        #feature3: Jaccard similarity
        jaccard_sim = get_jaccard_sim(tokens_chq,tokens_faq)
        vector1 = text_to_vector(sent_chq)
        vector2 = text_to_vector(sent_faq)
        #feature 4: cosine similarity
        cosine = get_cosine(vector1, vector2)ngramngramngramngramngramngram
        #feature 5: Levenshtein
        lev=levenshtein(sent_chq,sent_faq)
        #feature6: maximum value obtained
        doc1=nlp(sent_faq)
        doc2=nlp(sent_chq)

        similarity = doc1.similarity(doc2)
        average=(word_overlap+bigram_overlap+jaccard_sim+cosine+lev)/6
        question_length_ratio=len(sent_chq)/len(sent_faq)
        common_nouns=nouns(sent_chq,sent_faq)
        common_verbs=verbs(sent_chq,sent_faq)
        pair_arr = [word_overlap,bigram_overlap,jaccard_sim,cosine,lev,similarity,question_length_ratio,common_nouns,common_verbs]
        x_values.append(pair_arr)
        print(x_values)

tree = ET.parse('RQE_Train_8588_AMIA2016.xml')
y_values = []
x_values = []
training_set(tree,x_values,y_values)
x_frame = pd.DataFrame(np.row_stack(x_values))
y_frame = pd.DataFrame(np.row_stack(y_values))
gnb=GaussianNB()
gnb.fit(x_values,y_values)  
x_test = []
y_test = []
tree_test = ET.parse('RQE_Test_302_pairs_AMIA2016.xml')
training_set(tree_test,x_test,y_test)
predicted = gnb.predict(x_test)
print("Accuracy: ",accuracy_score(predicted,y_test)*100)