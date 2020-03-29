from typing import List, Union
import numpy as np
from networkx import DiGraph
from pandas import DataFrame, Series

from generics import GenericNode, GenericClass, GenericModel


class Id3Node(GenericNode):
    pass


class Id3Class(GenericClass, Id3Node):
    pass


class Id3Model(GenericModel[Id3Node]):
    def _browse_tree(self, node: Id3Node, attributes: Series):
        edges = self.tree.out_edges(node, 'value')
        candidates = [child for p, child, value in edges if value == attributes[node.name]]

        if len(candidates) == 0:
            raise AttributeError(
                "no candidate for {} in the given series with the value {}".format(node.name, attributes[node.name])
            )

        successor = candidates[0]

        if type(successor) == Id3Class:
            return successor.value

        return self._browse_tree(successor, attributes)


def generate_id3_tree(data: DataFrame):
    tree = DiGraph(directed=True)
    root = next_node(data, tree)
    return Id3Model(tree, root)


def next_node(data: DataFrame, tree: DiGraph) -> [Id3Node]:
    # for now we assume that there is data
    rows, cols_count = data.shape

    if rows == 0:  # impossible cas but for security
        return None
    if np.unique(data.iloc[:, -1].values).size == 1:
        return Id3Class(data.columns[-1], data.iloc[0, -1])

    # gather gains
    gains = [information_gain(data.to_numpy(), a) for a in range(0, cols_count-1)]
    # find the index of the column with best gain
    a = gains.index(max(gains))
    # retrieve column name from the column index
    column = data.columns[a]
    # get all possible values of the attribute and remove duplicates
    edges = set(data.iloc[:, a])
    # create a node for the actual attribute
    node = Id3Node(column, edges)
    tree.add_node(node)

    for edge in edges:
        subset = drop_attribute_value(data, column, edge)
        # Continue recursive generation of the tree
        child = next_node(subset, tree)
        node.child[edge] = child
        tree.add_edge(node, child, value=edge)

    return node


def drop_attribute_value(df, attribute, value):
    # Get index of rows that don't match the edge
    indexes = df[df[attribute] != value].index
    # Remove those rows
    subset = df.drop(indexes)
    # Remove attribute
    subset.drop(columns=attribute, inplace=True)
    return subset


def split_dataset(dataset: np.ndarray, index, as_dict=False) -> Union[dict, List[np.ndarray]]:
    """
    Extract subsets by splitting the dataset for each value of the attribute
    at the specified index. Either an attribute or the class
    :param dataset: The dataset to slit
    :param index: the index of the attribute to split upon (or -1 for the class)
    :param as_dict: Either to return a dict with the attribute value as key or the array without attributes value
    :return: subsets as np.ndarray
    :rtype: np.ndarray[] is as_dict equals to false, dict otherwise
    """
    classes = set(dataset[:, index])
    result = [dataset[dataset[:, index] == class_] for class_ in classes]
    if as_dict:
        return dict(zip(classes, result))
    else:
        return result


def entropy(dataset: np.ndarray):
    """
    Compute the entropy of the dataset based on a measure of the amount of uncertainty in the dataset.
    :param dataset: The dataset to analyse
    :return: float
    """
    rows, cols = dataset.shape
    classes_subsets = split_dataset(dataset, -1)
    proportions = [float(subset.shape[0]) / rows for subset in classes_subsets]
    return sum([-px * np.log2(px) for px in proportions])


def information_gain(dataset: np.ndarray, attribute):
    """
    Compute the information gain based on the measure of the difference in entropy from before to after the dataset is
    split on the attribute
    :param dataset: The dataset to analyse
    :param attribute: The attribute to split upon
    :return: float
    """
    rows, cols = dataset.shape
    subsets = split_dataset(dataset, attribute)
    h = entropy(dataset)
    return h - sum([float(subset.shape[0]) / rows * entropy(subset) for subset in subsets])
