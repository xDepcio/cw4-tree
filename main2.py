import pandas as pd
import numpy as np
from collections import Counter


def calculate_entropy(data):
    labels = data.iloc[:, -1]
    entropy = 0
    for label in np.unique(labels):
        probability = np.sum(labels == label) / len(labels)
        entropy -= probability * np.log2(probability)
    return entropy


def calculate_gain(data, attribute):
    total_entropy = calculate_entropy(data)
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / np.sum(counts))
        * calculate_entropy(data.where(data[attribute] == values[i]).dropna())
        for i in range(len(values))
    )
    return total_entropy - weighted_entropy


def get_most_informative_feature(data):
    feature_gains = {
        feature: calculate_gain(data, feature) for feature in data.columns[:-1]
    }
    return max(feature_gains, key=feature_gains.get)


def build_tree(data, tree=None):
    feature = get_most_informative_feature(data)
    if tree is None:
        tree = {}
        tree[feature] = {}
    for value in np.unique(data[feature]):
        sub_data = data.where(data[feature] == value).dropna()
        class_value, counts = np.unique(sub_data["PlayTennis"], return_counts=True)
        if len(counts) == 1:
            tree[feature][value] = class_value[0]
        else:
            tree[feature][value] = build_tree(sub_data)
    return tree


def classify(instance, tree):
    for nodes in tree.keys():
        value = instance[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = classify(instance, tree)
        else:
            prediction = tree
            break

    return prediction


def main():
    data = pd.read_csv("test-data.csv")
    # cut day column
    data = data.drop("Day", axis=1)
    tree = build_tree(data)
    print(tree)
    new_data_to_classify = pd.Series(
        {"Outlook": "Rain", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}
    )
    instance = data.iloc[0]  # Use the first row of the dataset as an example
    print(instance)
    print("Classification of the instance: ", classify(new_data_to_classify, tree))


if __name__ == "__main__":
    main()
