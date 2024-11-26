import nltk

ENTROPY_CUTOFF=0.6
SUPPORT_CUTOFF=6

def set_dtmodel_parameters(self):

    self.dtclassifier = \
        nltk.classify.DecisionTreeClassifier.train(self.training_data, 
                                                   entropy_cutoff=ENTROPY_CUTOFF, 
                                                   support_cutoff=SUPPORT_CUTOFF)
    
    self.dtclassifier_name = type(self.dtclassifier).__name__

    self.dttraining_set_accuracy = \
        nltk.classify.accuracy(self.dtclassifier, self.training_data)
    print('decision tree training set accuracy: ', self.dttraining_set_accuracy)

    self.dttest_set_accuracy = \
        nltk.classify.accuracy(self.dtclassifier, self.test_data)
    print('decision tree test set accuracy: ', self.dttest_set_accuracy)
