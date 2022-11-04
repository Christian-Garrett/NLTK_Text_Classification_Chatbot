import nltk 
import os
import csv

# nltk.download_gui()



class Preprocess:

    def __init__(self, filename):

        self.filename = filename
        self.text_data = self.get_content()


    def get_content(self):

        doc = os.path.join(self.filename)
        with open(doc, 'r') as content_file:
            lines = csv.reader(content_file,delimiter='|')
            file_data = [x for x in lines if len(x) == 3]

            return file_data