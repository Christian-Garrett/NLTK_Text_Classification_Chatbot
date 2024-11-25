import nltk

def set_dtmodel_parameters(self):

    self.dtclassifier = \
        nltk.classify.DecisionTreeClassifier.train(self.training_data, 
                                                   entropy_cutoff=0.6, 
                                                   support_cutoff=6)
    
    self.dtclassifier_name = type(self.dtclassifier).__name__

    self.dttraining_set_accuracy = \
        nltk.classify.accuracy(self.dtclassifier, self.training_data)
    print('decision tree training set accuracy: ', self.dttraining_set_accuracy)

    self.dttest_set_accuracy = \
        nltk.classify.accuracy(self.dtclassifier, self.test_data)
    print('decision tree test set accuracy: ', self.dttest_set_accuracy)
