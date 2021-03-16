
import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk import ClassifierI
import loadData
def removeFreqFromCommonWords(most_common_words):
    wordsCleaned=[]
    for w in most_common_words:
        wordsCleaned.append(w[0])
    return wordsCleaned

def find_features(document,common_words):
    '''returns a dict containing every word of the common_words as a key 
       and a boolean as a value indicating if the key appears in the document list. |example: { 'the'=True ],
                                                                                                'cool'=False}
       would indicate that the word the is in the document and the word cool is not'''
    documentWords = set(document)
    features = {}
    
    for word in common_words:
        features[word] = (word in documentWords)

    return features

class vote_classifier(ClassifierI):
    '''takes a list of classifiers and gives methods that find the most voted prediction and the confidence on it'''
    def __init__(self, classifiers):
        self._classifiers = classifiers
         
    def classify(self,features):
        '''returns the most voted answer from all the classifiers'''
        vote=[]
        for classifier in self._classifiers:
            prediction=classifier.classify(features)
            vote.append(prediction)
        MostVotedPrediction=mode(vote)
        return MostVotedPrediction
    
    def confidence(self,features):
        '''returns the confidence level of the prediction'''
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#uncommend this two lines  if you load the data from the corpus for the first time
# documents=loadData.getData('withoughtTheOs.csv')
# loadData.pickleData(documents)

with open('documents.pickle', 'rb') as file:
       documents=pickle.load(file)

random.shuffle(documents)

#get each and every word of every article in the database
all_words = []
for document in documents:
    for article in document :
        for word in article:
            all_words.append(word.lower())
all_words = nltk.FreqDist(all_words)

#I need to clean the data (lemmatize /remove stopwords ) 
most_common_words =all_words.most_common(5000)
most_common_words_Cleaned=removeFreqFromCommonWords(most_common_words)


#creating the features 
featuresets = [(find_features(rev,most_common_words_Cleaned), category) for (rev, category) in documents]


#give training set 80% of the corpus and testing set 20%
training_set=featuresets[:int(len(featuresets)*0.8)]
testing_set=featuresets[int(len(featuresets)*0.8):]



#training the classifiers (and put them in a pickle)


loadData.trainAlgos(training_set)

#loads the classifiers 
with open('classifiers.pickle', 'rb') as file:
    classifiers=pickle.load(file)

#print each classifiers accuracy : 
print("\nCalculating each classifiers accuracy on the testing set: ")
for x in classifiers:
    print(f"{x} accuracy percent:", (nltk.classify.accuracy(x, testing_set))*100)
print("------------------------------------------------------------- ")
print("\n Taking the consensus and calculate its accuracy... ")
votes=vote_classifier(classifiers)


#get the most voted guess and the confidence from all of the classifiers
# totalnegatives=0
# for features in testing_set:                                                                                     
#     mostVotedPrediction=votes.classify(features[0])
#     if(mostVotedPrediction=='neg'):
#         totalnegatives+=1    
#     confidence=votes.confidence(features[0])
# print(totalnegatives)
# print('from total articles:')
# print(len(testing_set))
    

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(votes, testing_set))*100)
print("------------------------------------------------------------- ")
