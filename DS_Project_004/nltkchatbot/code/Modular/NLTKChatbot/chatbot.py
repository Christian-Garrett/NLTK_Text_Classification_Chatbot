import numpy as np
import pickle
from pathlib import Path
import sys
import os

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from MLPipeline.ExtractFeatures import create_match_dict, extract_features


def chatbot_response(responses, dtclassifier, input_sentence):
    category = dtclassifier.classify(create_match_dict(extract_features(input_sentence)))

    return responses.item()[category]

##### todo: add in dynamic paths
response_dict = np.load('answer_dictionary.npy', allow_pickle=True)
loaded_model = pickle.load(open('working_dt_model.sav', 'rb'))

##### todo: use the input function to get user question and create a processing loop
Question = 'How many annual leaves do I have left?'

''' Remove:
# Question = 'How many annual leaves do I have left?'
# print(Question)
# '''
print(chatbot_response(response_dict, loaded_model, Question))

##### todo: add in main code guard
