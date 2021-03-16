import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
def getData(csvFileName):
    #return is a list of tuples [(list,str),(list,str)...]. Each tuple contains a list and a string (list,str) . 
    # The list is an article but every word is in separate cell. 
    # the string is 'neg'  'pos' or 'o' indicating if the review is negative or positive or neutral
    database=[]
    with open(csvFileName,'r',encoding="utf8") as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            sent=line[0]
            content=word_tokenize(line[2])
            database.append((content,sent))
    return database
   
        

def pickleData(documents):
    '''pickles the document for latter use'''
    with open('documents.pickle', 'wb') as file:
      pickle.dump(documents,file)


def trainAlgos(training_set):
    '''it trains every classifier (naive bayes bernoulliNB MultinomialNB LogisticRregretion SGDC svc lINEARSVC Nusvc)
    and saves them in a pickle named classifiers.pickle'''
    
    classifiers=[]
    print("------------------------------------------------------------- ")
    print("Training classifiers. Please wait...\n ")
    Naive_bayes_classifier=nltk.NaiveBayesClassifier.train(training_set)
    print("Naive bayes classifier is trained. (1/8) ")
    BNB_classifier =SklearnClassifier(BernoulliNB()).train(training_set)
    print("Bernoulli classifier is trained. (2/8)")
    MNB_classifier = SklearnClassifier(MultinomialNB()).train(training_set)
    print("Multinomial classifier is trained. (3/8)")
    LogisticRegression_classifier  = SklearnClassifier(LogisticRegression(max_iter=20000)).train(training_set)
    print("Logistic Regression classifier is trained. (4/8)")
    SGDClassifier_classifier  = SklearnClassifier(SGDClassifier()).train(training_set)
    print("SGD classifier is trained. (5/8)")
    SVC_classifier = SklearnClassifier(SVC()).train(training_set)
    print("SVC classifier is trained. (6/8)")
    LinearSVC_classifier =  SklearnClassifier(LinearSVC()).train(training_set)
    print("Linear SVC classifier  is trained. (7/8)")
    NuSVC_classifier = SklearnClassifier(NuSVC()).train(training_set)
    print("NuSVC classifier is trained. (8/8)")
    Naive_bayes_classifier.show_most_informative_features(50)
    print("------------------------------------------------------------- ")
    classifiers.extend([Naive_bayes_classifier,BNB_classifier,MNB_classifier,LogisticRegression_classifier,SGDClassifier_classifier,SVC_classifier,LinearSVC_classifier,NuSVC_classifier])
    with open('classifiers.pickle', 'wb') as file:
       pickle.dump(classifiers,file)

    return None