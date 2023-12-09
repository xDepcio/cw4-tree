import numpy as np
import pandas as pd


def load_csv_into_pandas():
    col_names = ["Day", "Outlook", "Temp", "Humidity", "Wind", "PlayTennis"]
    data = pd.read_csv("test-data.csv", skiprows=1, header=None, names=col_names)
    return data


def calculate_entropy(data):
    entropy = 0
    play_tennis = data.keys()[-1]
    values = data[play_tennis].unique()
    for value in values:
        fraction = data[play_tennis].value_counts()[value] / len(data[play_tennis])
        entropy += -fraction * np.log2(fraction)
    return entropy


def get_most_informative_feature(data):
    information_gain = []
    for key in data.keys()[:-1]:
        information_gain.append(caluclate_gain_for_attribute(data, key))
    return data.keys()[:-1][np.argmax(information_gain)]


def caluclate_gain_for_attribute(data, attribute):
    gain = calculate_entropy(data)
    attribute_values = data[attribute].unique()
    for attribute_value in attribute_values:
        sub_data = data[data[attribute] == attribute_value]
        gain -= len(sub_data) / len(data) * calculate_entropy(sub_data)
    return gain


def main():
    data = load_csv_into_pandas()
    # cut day column
    data = data.drop("Day", axis=1)
    print("Entropy of the dataset:", calculate_entropy(data))
    print("entropy of sunny day:", calculate_entropy(data[data["Outlook"] == "Sunny"]))
    print(
        "entropy of overcast day:",
        calculate_entropy(data[data["Outlook"] == "Overcast"]),
    )
    print("entropy of rainy day:", calculate_entropy(data[data["Outlook"] == "Rain"]))
    print(
        "calculating gain for outlook:", caluclate_gain_for_attribute(data, "Outlook")
    )
    print("Most informative feature:", get_most_informative_feature(data))


if __name__ == "__main__":
    main()
