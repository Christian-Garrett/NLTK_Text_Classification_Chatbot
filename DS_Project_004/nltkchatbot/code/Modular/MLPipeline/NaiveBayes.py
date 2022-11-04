import nltk
from nltk.classify import SklearnClassifier


class NaiveBayes:

    def __init__(self, train_data, test_data):

        self.train_data = train_data
        self.test_data = test_data

        self.classifier = [] 
        self.classifier_name = [] 
        self.test_set_accuracy = 0.0
        self.training_set_accuracy = 0.0

        self.train_using_naive_bayes()
        

    def train_using_naive_bayes(self): 
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_data)
        self.classifier_name = type(self.classifier).__name__
        self.training_set_accuracy = nltk.classify.accuracy(self.classifier, self.train_data)
        print('naive bayes training set accuracy: ', self.training_set_accuracy)
        self.test_set_accuracy = nltk.classify.accuracy(self.classifier, self.test_data)
        print('naive bayes test set accuracy: ', self.test_set_accuracy)


    def get_info(self):
        return self.classifier, self.classifier_name, self.test_set_accuracy, self.training_set_accuracy

