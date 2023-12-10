from typing import Dict, List, Union
import pandas as pd
import numpy as np
from collections import Counter


class Node:
    def __init__(self, most_common_attribute_choice: str):
        self.decision_attribute: str = ""
        self.decision_attribute_choices: Dict[str, Union[Node, str]] = {}
        self.most_common_attribute_choice = most_common_attribute_choice

    def add_child(self, child_name: str, child: Union["Node", str]):
        self.decision_attribute_choices[child_name] = child


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


# type TreeType = Dict[str, Union["TreeType", str]]


def build_tree(data: pd.DataFrame, tree: Node | None = None) -> Node:
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
        tree = Node(Counter(data[data.columns[-1]]).most_common(1)[0][0])

    attr_unique_values = np.unique(data[attribute])
    for value in attr_unique_values:
        sub_data = data.where(data[attribute] == value).dropna()
        classes_col = sub_data[sub_data.columns[-1]]  # Get the classes column
        class_value, counts = np.unique(classes_col, return_counts=True)
        if len(counts) == 1:
            tree.decision_attribute = attribute
            tree.add_child(value, class_value[0])
        else:
            tree.decision_attribute = attribute
            tree.add_child(value, build_tree(sub_data))

    return tree


def classify(instance: pd.Series, tree: Node):
    if isinstance(tree, str):
        return tree

    attribute = tree.decision_attribute
    attribute_value = instance[attribute]
    if attribute_value not in tree.decision_attribute_choices:
        return tree.most_common_attribute_choice
    return classify(instance, tree.decision_attribute_choices[attribute_value])


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


def calculate_accuracy(test_data: pd.DataFrame, tree: Node):
    correct = 0
    for row in test_data.iloc:
        if classify(row, tree) == row[-1]:
            correct += 1
    return correct / len(test_data)


def main():
    data: pd.DataFrame = load_data_frame(
        path="data/mushroom/agaricus-lepiota.data",
        class_col="Edibility",
        col_names=[
            "Edibility",
            "Cap-shape",
            "Cap-surface",
            "Cap-color",
            "Bruises?",
            "Odor",
            "Gill-attachment",
            "Gill-spacing",
            "Gill-size",
            "Gill-color",
            "Stalk-shape",
            "Stalk-root",
            "Stalk-surface-above-ring",
            "Stalk-surface-below-ring",
            "Stalk-color-above-ring",
            "Stalk-color-below-ring",
            "Veil-type",
            "Veil-color",
            "Ring-number",
            "Ring-type",
            "Spore-print-color",
            "Population",
            "Habitat",
            # "Class",
            # "age",
            # "menopause",
            # "tumor-size",
            # "inv-nodes",
            # "node-caps",
            # "deg-malig",
            # "breast",
            # "breast-quad",
            # "irradiat",
        ],
    )
    train_data, test_data = split_train_data(data, train_size=3 / 5)
    # data = data.drop("Day", axis=1)
    tree = build_tree(train_data)
    print(tree)
    accuracy = calculate_accuracy(test_data, tree)
    print("Accuracy: ", accuracy)

    instance = data.iloc[1]  # Use the first row of the dataset as an example
    instance = instance.drop("Edibility")
    print(instance)
    print("Classification of the instance: ", classify(instance, tree))


if __name__ == "__main__":
    main()
