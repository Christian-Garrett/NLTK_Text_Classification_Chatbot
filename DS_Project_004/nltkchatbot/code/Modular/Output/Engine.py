from pathlib import Path
import os
import sys

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from MLPipeline import BuildDataset
from MLPipeline import TrainModels


def create_training_data():

    input_data_path = os.path.join(module_path, "Input/leaves.txt")

    dataset_object = BuildDataset(input_data_path)
    dataset_object.preprocess_data()
    # dataset_object.save_processed_data()


def train_chatbot():

    training_data_path = os.path.join(module_path, "Input/training_data.npy")
    testing_data_path = os.path.join(module_path, "Input/test_data.npy")
    tc_object = TrainModels(training_data_path, testing_data_path)
    tc_object.train_decision_tree_model()
    tc_object.train_naive_bayes_model()
    # tc_object.save_chatbot_model()


create_training_data()
train_chatbot()
