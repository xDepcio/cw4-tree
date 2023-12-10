from typing import Dict, List, Union
import pandas as pd
import numpy as np
from collections import Counter


def calculate_entropy(data: pd.DataFrame):
    class_col = data.iloc[:, -1]
    entropy = 0
    for unique_class_val in np.unique(class_col):
        probability = np.sum(class_col == unique_class_val) / len(class_col)
        entropy -= probability * np.log2(probability)
    return entropy


def calculate_gain(data: pd.DataFrame, attribute: str):
    total_entropy = calculate_entropy(data)
    values, counts = np.unique(data[attribute], return_counts=True)
    values_entropy = sum(
        (counts[i] / np.sum(counts))
        * calculate_entropy(data.where(data[attribute] == values[i]).dropna())
        for i in range(len(values))
    )
    return total_entropy - values_entropy


def get_most_informative_attribute(data: pd.DataFrame) -> str:
    attribute_gains = [
        (attribute, calculate_gain(data, attribute)) for attribute in data.columns[:-1]
    ]

    return max(attribute_gains, key=lambda x: x[1])[0]


type TreeType = Dict[str, Union["TreeType", str]]


def build_tree(data: pd.DataFrame, tree: TreeType | None = None) -> TreeType:
    # Get the feature with the highest information gain
    attribute = get_most_informative_attribute(data)

    # If all attrubites have only one value, return the most common class
    each_col_ony_one_val = [
        data[col_name].nunique() == 1 for col_name in data.columns[:-1]
    ]
    if all(each_col_ony_one_val):
        # Counter - buffed dictionary
        return Counter(data[data.columns[-1]]).most_common(1)[0][0]

    # If class col has only one value, return it
    if len(np.unique(data[data.columns[-1]])) == 1:
        return np.unique(data[data.columns[-1]])[0]

    # Continue building the tree
    if tree is None:
        tree = {}
        tree[attribute] = {}

    attr_unique_values = np.unique(data[attribute])
    for value in attr_unique_values:
        sub_data = data.where(data[attribute] == value).dropna()
        classes_col = sub_data[sub_data.columns[-1]]  # Get the classes column
        class_value, counts = np.unique(classes_col, return_counts=True)
        if len(counts) == 1:
            tree[attribute][value] = class_value[0]
        else:
            tree[attribute][value] = build_tree(sub_data)

    return tree


def classify(instance: pd.Series, tree: TreeType):
    prediction = 0
    for key in tree.keys():
        value = instance[key]
        tree = tree[key][value]

        if type(tree) is dict:
            prediction = classify(instance, tree)
        else:
            prediction = tree
            break

    return prediction


def load_data_frame(
    path: str,
    class_col: str,
    col_names: List[str],
    skiprows: int = 0,
    cut_cols: List[str] = [],
) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=skiprows, names=col_names)

    cols = list(data.columns.values)
    cols.pop(cols.index(class_col))
    data = data[cols + [class_col]]

    for col in cut_cols:
        data = data.drop(col, axis=1)
    return data


def split_train_data(
    data: pd.DataFrame, train_size: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for row in data.iloc:
        if np.random.rand() < train_size:
            train_data = pd.concat([train_data, pd.DataFrame([row])])
        else:
            test_data = pd.concat([test_data, pd.DataFrame([row])])

    return train_data, test_data


def calculate_accuracy(test_data: pd.DataFrame, tree: TreeType):
    correct = 0
    for row in test_data.iloc:
        if classify(row, tree) == row[-1]:
            correct += 1
    return correct / len(test_data)


def main():
    data: pd.DataFrame = load_data_frame(
        path="data/breast-cancer.data",
        class_col="Class",
        col_names=[
            "Class",
            "age",
            "menopause",
            "tumor-size",
            "inv-nodes",
            "node-caps",
            "deg-malig",
            "breast",
            "breast-quad",
            "irradiat",
        ],
    )
    train_data, test_data = split_train_data(data, train_size=3 / 5)
    # cut day column
    # data = data.drop("Day", axis=1)
    tree = build_tree(train_data)
    print(tree)
    accuracy = calculate_accuracy(test_data, tree)
    print("Accuracy: ", accuracy)
    # new_data_to_classify = pd.Series(
    #     {"Outlook": "Rain", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}
    # )
    instance = data.iloc[1]  # Use the first row of the dataset as an example
    instance = instance.drop("Class")
    print(instance)
    print("Classification of the instance: ", classify(instance, tree))


if __name__ == "__main__":
    main()
