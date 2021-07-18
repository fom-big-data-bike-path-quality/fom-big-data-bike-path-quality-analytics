from sklearn.model_selection import train_test_split


#
# Main
#


class TrainTestDataSplitter:

    def run(self, dataframes, test_size=0.15, random_state=0):
        ids = sorted(list(dataframes.keys()))
        train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)

        train_dataframes = {id: dataframes[id] for id in train_ids}
        test_dataframes = {id: dataframes[id] for id in test_ids}

        return train_dataframes, test_dataframes
