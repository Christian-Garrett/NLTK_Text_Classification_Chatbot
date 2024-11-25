import nltk
# nltk.download('omw-1.4')
# nltk.download_gui()

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def save_model_training_components(self):
    return None


def create_match_dict(words):
    return dict([(word, True) for word in words])


# filter out processed words that do not have 'relevant' tags
def extract_tagged(sentences):

    features = []
    for tagged_word in sentences:
        word, tag = tagged_word
        if tag =='NN' or tag == 'VBN' or tag == 'NNS' or \
            tag == 'VBP' or tag == 'RB' or tag == 'VBZ' or \
            tag == 'VBG' or tag =='PRP' or tag == 'JJ':
            features.append(word)

    return features


# create a lowercase list and remove stop words 
def process_text(sentence):

    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = \
        [word for word in tokens 
         if not word in stopwords.words('english')]

    return filtered_words


# standardize the input data for relevant features
def extract_features(text):
    processed_words = process_text(text)
    tagged_words = nltk.pos_tag(processed_words)
    extracted_words = extract_tagged(tagged_words)

    stemmer = SnowballStemmer("english")
    stemmed_words = \
        [stemmer.stem(word) for word in extracted_words]

    lmtzr = WordNetLemmatizer()
    result = [lmtzr.lemmatize(word) for word in stemmed_words]

    return result


def get_model_training_components(self):

    corpus_update = [] # the corpus of lemmatized words
    training_example_update = [] # category match supervised learning example
    answer_dict_update = {}  # responses from the chat bot
    for (text, category, answer) in self.input_text_data:

        lemmatized_words = extract_features(text)

        corpus_update.append(lemmatized_words)
        training_example_update.append([create_match_dict(lemmatized_words), category])
        answer_dict_update[category] = answer

        self.training_examples = training_example_update 
        self.corpus_list = sum(corpus_update, [])
        self.answers_dict = answer_dict_update
