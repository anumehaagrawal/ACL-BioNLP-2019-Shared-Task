import pandas as pd 
import xml.etree.ElementTree as ET
import nltk
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
from nltk.corpus import wordnet as wn
from nltk.tag import StanfordNERTagger
#word_vectors = api.load("glove-wiki-gigaword-100")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import svm
from sklearn.metrics import accuracy_score
st = StanfordNERTagger('/home/anumeha/Documents/ACL-BioNLP-2019-Shared-Task/RQE/english.all.3class.distsim.crf.ser.gz','/home/anumeha/Documents/ACL-BioNLP-2019-Shared-Task/RQE/stanford-ner.jar',
encoding='utf-8')

def cleaned_words(myString):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    stemmer = PorterStemmer()
    result = [' '.join([stemmer.stem(w).lower() for w in x.split()]) for x in tokens if x.lower() not in stopwords.words('english')]
    return result
    
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return len(lst3) 

def get_jaccard_sim(a, b): 
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score['neg'],score['pos']

def count_named_entities(text):
        tokenized_text = word_tokenize(text)
        classified_text = st.tag(tokenized_text)
        count=0

        for i in classified_text:
                if i[1]!='O':
                        count+=1

        return count

def training_set(tree,x_values,y_values):
    
    root = tree.getroot()
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
        word_overlap = intersection(tokens_chq,tokens_faq)/len(tokens_chq)
        bigram_chq = list(ngrams(tokens_chq, 2)) 
        bigram_faq = list(ngrams(tokens_faq, 2)) 
        bigram_overlap = intersection(bigram_chq,bigram_faq)/len(bigram_chq)
        jaccard_sim = get_jaccard_sim(tokens_chq,tokens_faq)
        #similarity = word_vectors.wmdistance(tokens_chq, tokens_faq)
        neg_chq,pos_chq = sentiment_analyzer_scores(chq)
        neg_faq,pos_faq = sentiment_analyzer_scores(faq)
        neg_val = neg_chq*neg_faq
        pos_val = pos_faq*pos_chq
        chq_ner = count_named_entities(chq)/len(tokens_chq)
        faq_ner = count_named_entities(faq)/len(tokens_faq)
        pair_arr = [word_overlap,bigram_overlap,jaccard_sim,neg_val,pos_val,chq_ner,faq_ner]
        x_values.append(pair_arr)
        

tree = ET.parse('train.xml')
y_values = []
x_values = []
training_set(tree,x_values,y_values)
x_frame = pd.DataFrame(np.row_stack(x_values))
y_frame = pd.DataFrame(np.row_stack(y_values))
clf = svm.SVC(gamma='scale')
clf.fit(x_values,y_values)  
x_test = []
y_test = []
tree_test = ET.parse('test.xml')
training_set(tree_test,x_test,y_test)
predicted = clf.predict(x_test)
print(accuracy_score(predicted,y_test))
