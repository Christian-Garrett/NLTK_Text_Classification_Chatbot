import csv


def get_text_data(self):

    with open(self.file_name, 'r') as content_file:
        file_data = csv.reader(content_file, delimiter='|')
        sample_data = \
            [sample for sample in file_data if len(sample) == 3]

    return sample_data
