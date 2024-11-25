import nltk


def set_nbmodel_parameters(self):

    self.nbclassifier = nltk.NaiveBayesClassifier.train(self.training_data)
    self.nbclassifier_name = type(self.nbclassifier).__name__

    self.nbtraining_set_accuracy = nltk.classify.accuracy(self.nbclassifier, self.training_data)
    print('naive bayes training set accuracy: ', self.nbtraining_set_accuracy)

    self.nbtest_set_accuracy = nltk.classify.accuracy(self.nbclassifier, self.test_data)
    print('naive bayes test set accuracy: ', self.nbtest_set_accuracy)
