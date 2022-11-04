import numpy as np
import random
import pickle
from nltk.classify import SklearnClassifier


from nltkchatbot.code.Modular.MLPipeline.Preprocess import Preprocess
from nltkchatbot.code.Modular.MLPipeline.ExtractFeatures import ExtractFeatures
from nltkchatbot.code.Modular.MLPipeline.DecisionTree import DecisionTree
from nltkchatbot.code.Modular.MLPipeline.NaiveBayes import NaiveBayes


file_path = "nltkchatbot\code\Modular\Input\leaves.txt"


def chat_bot(extract, responses, dtclassifier, input_sentence):
    category = dtclassifier.classify(extract.word_feats(extract.extract_feature(input_sentence)))

    return responses.item()[category]


def split_dataset(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)

    return (data[:train_split]), (data[train_split:])


def train(file_path):

    input_file_object = Preprocess(file_path)

    # raw data list format: [sentence, category, response]
    raw_data = input_file_object.text_data    

    features_object = ExtractFeatures(raw_data)
    features_data, corpus, answers = features_object.get_info()

    # save the responses to an output file
    np.save('answer_dictionary', answers)

    # split data into train and test sets
    split_ratio = 0.8

    training_data, test_data = split_dataset(features_data, split_ratio)

    # save the data
    np.save('training_data', training_data)
    np.save('test_data', test_data)
    

def main():

    # load the data
    training_data = np.load('training_data.npy', allow_pickle=True)
    test_data = np.load('test_data.npy', allow_pickle=True)

    # train decision tree model
    decision_tree = DecisionTree(training_data, test_data)
    dtclassifier, dtclassifier_name, dttest_set_accuracy, dttraining_set_accuracy = decision_tree.get_info()

    # train naive bayes model
    naive_bayes = NaiveBayes(training_data, test_data)
    nbclassifier, nbclassifier_name, nbtest_set_accuracy, nbtraining_set_accuracy = naive_bayes.get_info()
    print(len(nbclassifier.most_informative_features()))
    nbclassifier.show_most_informative_features()

    # save the model to an output file
    filename = 'working_dt_model.sav'
    pickle.dump(dtclassifier, open(filename, 'wb'))


# train(file_path)

# main()

## load the trained decision tree model and the response dictionary from disk
extraction_object = ExtractFeatures([['test', 'test', 'test']])
response_dict = np.load('answer_dictionary.npy', allow_pickle=True)
loaded_model = pickle.load(open('working_dt_model.sav', 'rb'))

Question = 'How many annual leaves do I have left?'
print(Question)
print(chat_bot(extraction_object, response_dict, loaded_model, Question))

