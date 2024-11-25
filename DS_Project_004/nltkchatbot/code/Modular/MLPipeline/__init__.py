from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np


module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

##### todo: add class documentation 

class BuildDataset:

    from MLPipeline.LoadData import get_text_data
    from MLPipeline.ExtractFeatures import get_model_training_components
    from MLPipeline.SaveData import save_model_training_components

    def __init__(self, filename, output_path="Output/"):

        self.output_path=os.path.join(module_path, output_path)
        self.filename=filename
        self.input_text_data=self.get_text_data(filename)
        self.training_examples=None
        self.corpus_list=None
        self.answer_dict=None

    def preprocess_data(self):
        self.get_model_training_components()

    def save_processed_data(self):
        self.save_model_training_components()


class TrainModels:

    from MLPipeline.DecisionTree import set_dtmodel_parameters
    from MLPipeline.NaiveBayes import set_nbmodel_parameters
    from MLPipeline.SaveData import save_chatbot_model

    def __init__(self, training_data_path, test_data_path, output_path="Output/"):
        self.output_path=os.path.join(module_path, output_path)
        self.training_data_path=training_data_path
        self.test_data_path=test_data_path
        self.training_data=\
            np.load(self.training_data_path, allow_pickle=True)
        self.test_data=\
            np.load(self.test_data_path, allow_pickle=True)
        self.dtclassifier=None
        self.dtclassifier_name=None
        self.dttest_set_accuracy=None
        self.dttraining_set_accuracy=None
        self.nbclassifier=None,
        self.nbclassifier_name=None,
        self.nbtest_set_accuracy=None,
        self.nbtraining_set_accuracy=None

    def train_decision_tree_model(self):
        self.set_dtmodel_parameters()

    def train_naive_bayes_model(self):
        self.set_nbmodel_parameters()

    def save_chatbot_model(self):
        self.save_model()
