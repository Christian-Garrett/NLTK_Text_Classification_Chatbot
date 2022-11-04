import nltk
from nltk.classify import SklearnClassifier



class DecisionTree:

    def __init__(self, train_data, test_data):

        self.train_data = train_data
        self.test_data = test_data

        self.dtclassifier = [] 
        self.classifier_name = [] 
        self.test_set_accuracy = 0.0
        self.training_set_accuracy = 0.0

        self.train_using_decision_tree()

    
    def train_using_decision_tree(self):
    
        self.dtclassifier = nltk.classify.DecisionTreeClassifier.train(self.train_data, entropy_cutoff=0.6, support_cutoff=6)
        self.classifier_name = type(self.dtclassifier).__name__
        self.training_set_accuracy = nltk.classify.accuracy(self.dtclassifier, self.train_data)
        print('decision tree training set accuracy: ', self.training_set_accuracy)
        self.test_set_accuracy = nltk.classify.accuracy(self.dtclassifier, self.test_data)
        print('decision tree test set accuracy: ', self.test_set_accuracy)


    def get_info(self):
        return self.dtclassifier, self.classifier_name, self.test_set_accuracy, self.training_set_accuracy