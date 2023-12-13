from typing import Dict, List, Union
import pandas as pd
import numpy as np
from collections import Counter
import numpy.typing as npt

from utils import calculate_entropy_u, load_data_frame_u, split_train_data_u


class Node:
    def __init__(self, most_common_attribute_choice: str):
        self.decision_attribute_idx: int = -1
        self.decision_attribute_choices: Dict[str, Union[Node, str]] = {}
        self.most_common_attribute_choice = most_common_attribute_choice

    def add_child(self, child_name: str, child: Union["Node", str]):
        self.decision_attribute_choices[child_name] = child


class Tree:
    def __init__(self, data_path: str):
        self.data = load_data_frame_u(data_path)
        train_data, test_data = split_train_data_u(self.data, train_size=3 / 5)
        self.test_data = test_data
        self.root = self._build_tree(train_data)
        self.accuracy = self._calculate_accuracy(test_data, self.root)

    def classify(self, instance: pd.Series, tree: Node):
        if isinstance(tree, str):
            return tree

        attribute_idx = tree.decision_attribute_idx
        attribute_value = instance[attribute_idx]
        if attribute_value not in tree.decision_attribute_choices:
            return tree.most_common_attribute_choice
        return self.classify(
            instance, tree.decision_attribute_choices[attribute_value]  # type: ignore
        )

    def calculate_confusion_matrix(
        self, test_data: pd.DataFrame, positive_class: str, negative_class: str
    ):
        """(TP, TN, FP, FN)"""
        tp, tn, fp, fn = 0, 0, 0, 0

        for row in test_data.iloc:
            if self.classify(row, self.root) == positive_class:
                if row.iloc[0] == positive_class:
                    tp += 1
                else:
                    fp += 1
            else:
                if row.iloc[0] == negative_class:
                    tn += 1
                else:
                    fn += 1

        return tp, tn, fp, fn

    def _calculate_gain(self, data: pd.DataFrame, attribute: str):
        total_entropy = calculate_entropy_u(data)
        values, counts = np.unique(data[attribute], return_counts=True)
        values_entropy = sum(
            (counts[i] / np.sum(counts))
            * calculate_entropy_u(data.where(data[attribute] == values[i]).dropna())
            for i in range(len(values))
        )
        return total_entropy - values_entropy

    def _calculate_accuracy(self, test_data: pd.DataFrame, tree: Node):
        correct = 0
        for row in test_data.iloc:
            if self.classify(row, tree) == row.iloc[0]:
                correct += 1
        return correct / len(test_data)

    def _get_most_informative_attribute_idx(self, data: pd.DataFrame) -> int:
        attribute_gains = [
            (attribute, self._calculate_gain(data, attribute))
            for attribute in data.columns[1:]
        ]
        max_val = max(attribute_gains, key=lambda x: x[1])
        return int(max_val[0])

    def _build_tree(self, data: pd.DataFrame, tree: Node | None = None):
        # If all attrubites have only one value, return the most common class
        each_col_ony_one_val = [
            data[col_name].nunique() == 1 for col_name in data.columns[1:]
        ]
        if all(each_col_ony_one_val):
            # Counter - buffed dictionary
            return Counter(data[data.columns[0]]).most_common(1)[0][0]

        # If class col has only one value, return it
        if len(np.unique(data[data.columns[0]])) == 1:
            return np.unique(data[data.columns[0]])[0]

        # Get the feature with the highest information gain
        attribute_idx = self._get_most_informative_attribute_idx(data)

        # Continue building the tree
        if tree is None:
            tree = Node(Counter(data[0]).most_common(1)[0][0])

        attr_unique_values = np.unique(data[attribute_idx])
        for value in attr_unique_values:
            sub_data = data.where(data[attribute_idx] == value).dropna()
            classes_col = sub_data[sub_data.columns[0]]  # Get the classes column
            class_value, counts = np.unique(classes_col, return_counts=True)
            if len(counts) == 1:
                tree.decision_attribute_idx = attribute_idx
                tree.add_child(value, class_value[0])
            else:
                tree.decision_attribute_idx = attribute_idx
                sub_data_new = sub_data.iloc[:, sub_data.columns != attribute_idx]
                tree.add_child(value, self._build_tree(sub_data_new))

        return tree


def main():
    total_acc1 = 0
    all_confs1 = []
    for _ in range(20):
        treeBreastCancer = Tree("data/breast-cancer/breast-cancer.data")
        acc1 = treeBreastCancer.accuracy
        conf1 = treeBreastCancer.calculate_confusion_matrix(
            test_data=treeBreastCancer.test_data,
            positive_class="recurrence-events",
            negative_class="no-recurrence-events",
        )
        total_acc1 += acc1
        all_confs1.append(conf1)

    avg_acc1 = total_acc1 / 20
    avg_conf1 = np.mean(all_confs1, axis=0)

    total_acc2 = 0
    all_confs2 = []
    for _ in range(20):
        treeMushroom = Tree("data/mushroom/agaricus-lepiota.data")
        acc2 = treeMushroom.accuracy
        conf2 = treeMushroom.calculate_confusion_matrix(
            test_data=treeMushroom.test_data,
            positive_class="e",
            negative_class="p",
        )
        total_acc2 += acc2
        all_confs2.append(conf2)

    avg_acc2 = total_acc2 / 20
    avg_conf2 = np.mean(all_confs2, axis=0)

    print(
        f"Breast Cancer Accuracy: {avg_acc1}, Confusion Matrix: {avg_conf1}\nMustroom Accuracy: {avg_acc2}, Confusion Matrix: {avg_conf2}"
    )


# def main2():
#     data = pd.read_csv("data/breast-cancer/breast-cancer.data", header=None)
#     new_data = data.drop_duplicates(subset=data.columns[:-1], keep=False)
#     new_data.to_csv("data/breast-cancer/breast-cancer-new.data", index=False)


if __name__ == "__main__":
    main()
