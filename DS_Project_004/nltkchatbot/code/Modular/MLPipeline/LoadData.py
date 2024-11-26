import csv

NUM_INPUT_COLUMNS = 3


def get_text_data(self):

    with open(self.input_path, 'r') as content_file:
        file_data = csv.reader(content_file, delimiter='|')
        sample_data = \
            [sample for sample in file_data 
             if len(sample) == NUM_INPUT_COLUMNS]

    return sample_data
