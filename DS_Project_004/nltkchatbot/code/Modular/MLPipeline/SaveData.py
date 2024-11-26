from pathlib import Path
import numpy as np
import sys
import os
import random
import pickle

SPLIT_RATIO=0.8

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


def split_dataset(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)

    return (data[:train_split]), (data[train_split:])


def save_model_training_resources(self):
      
    np.save(self.output_path, self.answer_dict)

    split_ratio = SPLIT_RATIO
    training_data, test_data = \
        split_dataset(self.training_examples, split_ratio)

    np.save(self.output_path, training_data)
    np.save(self.output_path, test_data)


def save_chatbot_model(self):

    filename = \
        os.path.join(module_path, 'Output/working_dt_model.sav')
    pickle.dump(self.dtclassifier, open(filename, 'wb'))
