import re
import nltk
# nltk.download('omw-1.4')

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


class ExtractFeatures:

    def __init__(self, data):

        self.data = data
        self.features_data = []
        self.filtered_words = []
        self.corpus = [] 
        self.answers_dict = {}

        self.extract_feature_from_doc()

    def word_feats(self, words):
        return dict([(word, True) for word in words])

    # create a lowercase list and remove stop words 
    def pre_process(self, sentence):
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        filtered_words = [w for w in tokens if not w in stopwords.words('english')]

        return filtered_words

    # filter out processed words that do not have 'relevant' tags
    def extract_tagged(self, sentences):
        features = []
        for tagged_word in sentences:
            word, tag = tagged_word
            if tag=='NN' or tag == 'VBN' or tag == 'NNS' or tag == 'VBP' or tag == 'RB' or tag == 'VBZ' or tag == 'VBG' or tag =='PRP' or tag == 'JJ':
                features.append(word)

        return features

    # standardize the input data for relevant features
    def extract_feature(self, text):
        # print('words: ',words)
        processed_words = self.pre_process(text)
        # print('tags: ',tags)
        tagged_words = nltk.pos_tag(processed_words)
        # print('Extracted features: ',extracted_features)
        extracted_features = self.extract_tagged(tagged_words)
        # print(stemmed_words)
        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(x) for x in extracted_features]  
        # print(result)
        lmtzr = WordNetLemmatizer()
        result = [lmtzr.lemmatize(x) for x in stemmed_words] 

        return result
    

    def extract_feature_from_doc(self):
        corpus_adder = [] 
        temp_result = [] 

        # The responses from the chat bot
        answer_dict = {}
        for (text,category,answer) in self.data:

            features = self.extract_feature(text)

            corpus_adder.append(features)
            temp_result.append([self.word_feats(features), category])
            answer_dict[category] = answer

            self.features_data = temp_result
            self.corpus = sum(corpus_adder, [])
            self.answers_dict = answer_dict


    def get_info(self):
        return self.features_data, self.corpus, self.answers_dict



