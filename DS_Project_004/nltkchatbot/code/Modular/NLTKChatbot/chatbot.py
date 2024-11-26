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


def load_chatbot():

    response_dict_path = os.path.join(module_path, "Output/answer_dictionary.npy")
    response_dict = np.load(response_dict_path, allow_pickle=True)

    nltk_model_path = os.path.join(module_path, "Output/working_dt_model.sav")
    chatbot_model = pickle.load(open(nltk_model_path, 'rb'))

    return response_dict, chatbot_model


def run_chatbot(answer_key, chat_mod):

    while True:
        user_input = input("Enter your question (type 'exit' to quit): ")
        if user_input.lower() == 'exit': 
            print("Goodbye!") 
            break
        else:
            print(chatbot_response(answer_key, chat_mod, user_input))


if __name__ == '__main__':
    answers, model = load_chatbot()
    run_chatbot(answers, model)
