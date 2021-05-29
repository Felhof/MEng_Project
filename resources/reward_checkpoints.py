import csv


class CheckpointResults:

    def __init__(self, checkpoints=None, model_name="", data_directory="data/"):
        self.results = []
        self.checkpoints = checkpoints if checkpoints is not None else [4, 8, 12, 16, 20]
        self.model_name = model_name
        self.data_directory = data_directory

    def add_result(self, result):
        self.results.append(result)

    def save_results(self):
        location = self.data_directory + '{}.csv'.format(self.model_name)
        with open(location, mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(self.checkpoints)
            for result in self.results:
                results_writer.writerow(result)
