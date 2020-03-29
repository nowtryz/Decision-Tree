from typing import Union

import numpy as np
from networkx import DiGraph
from pandas import Series, DataFrame

from generics import GenericClass, GenericNode, GenericModel
from id3 import split_dataset, entropy, drop_attribute_value


class C45Node(GenericNode):

    def get_edge(self, attr_value):
        return attr_value


class C45NonDiscreetAttribute(C45Node):
    def __init__(self, name, threshold):
        super().__init__(name, self.create_edges(threshold))
        self.threshold = threshold

    @staticmethod
    def create_edges(threshold):
        return '<=' + threshold, '>' + threshold

    def get_edge(self, attr_value):
        if attr_value <= self.threshold:
            return self.edges[0]
        else:
            return self.edges[1]


class C45Class(GenericClass, C45Node):
    pass


class C45Model(GenericModel[C45Node]):
    def _browse_tree(self, node: C45Node, attributes: Series):
        edges = self.tree.out_edges(node, 'value')
        edge = node.get_edge(attributes[node.name])
        candidates = [child for p, child, value in edges if value == edge]

        if len(candidates) == 0:
            raise AttributeError(
                "no candidate for {} in the given series with the value {}".format(node.name, attributes[node.name])
            )

        successor = candidates[0]

        if type(successor) == C45Class:
            return successor.value

        return self._browse_tree(successor, attributes)


def generate_c4_5_tree(data: DataFrame):
    tree = DiGraph(directed=True)
    root = next_node(data, tree)
    print(tree)
    return C45Model(tree, root)


def next_node(data: DataFrame, tree: DiGraph) -> [C45Node]:
    # for now on we assume that there is data
    rows, cols_count = data.shape
    dataset = data.to_numpy()

    if rows == 0:  # impossible cas but for security
        return None
    if np.unique(data.iloc[:, -1].values).size == 1:
        return C45Class(data.columns[-1], data.iloc[0, -1])

    # gather gains
    computed = [compute_gain(data, a) for a in range(0, cols_count-1)]
    gains = [info['gain'] for info in computed]
    # find the index of the column with best gain
    a = gains.index(max(gains))
    # retrieve column name from the column index
    column = data.columns[a]
    gain_info = computed[a]

    # create a NonDiscreetAttribute if the attribute isn't discreet, trivial
    if gain_info['discreet']:
        # get all possible values of the attribute and remove duplicates
        edges = set(data.iloc[:, a])
        # create a node for the actual attribute
        node = C45Node(column, edges)
    else:
        node = C45NonDiscreetAttribute(column, gain_info['threshold'])
        edges = node.edges

    tree.add_node(node)

    for edge in edges:
        subset = drop_attribute_value(data, column, edge)
        # Continue recursive generation of the tree
        child = next_node(subset, tree)
        node.child[edge] = child
        tree.add_edge(node, child, value=edge)

    return node


def is_discreet(attribute: str, df: DataFrame):
    return np.issubdtype(df[attribute].dtype, np.number)


def compute_gain(df: DataFrame, attribute: int):
    dataset = df.to_numpy()
    if is_discreet(df.columns[attribute], df):
        return {
            'gain': information_gain(dataset, split_dataset(dataset, attribute)),
            'discreet': False
        }
    else:
        threshold, gain = best_threshold(dataset, attribute)
        return {
            'gain': gain,
            'discreet': True,
            'threshold': threshold
        }


def e(subset: np.ndarray, classes_with_na: dict, dataset_rows: int):
    classes_subsets = split_dataset(subset, -1, as_dict=True)
    # ratio_i = (pi + ni) / (Σi pi + ni) = (pi + ni) / (p + n)
    ratio = subset.shape[0] / dataset_rows
    # real pi = pi + pu * ratio
    values = [len(values) + len(classes_with_na.get(c, [])) * ratio for c, values in classes_subsets.items()]
    values_sum = sum(values)
    # prop = p / (p + n)
    probabilities = [v / values_sum for v in values]

    _entropy = - sum([p * np.log2(p) for p in probabilities])

    return float(values_sum) / dataset_rows * _entropy


def information_gain(dataset: np.ndarray, subsets):
    """
    Compute the information gain based on the measure of the difference in entropy from before to after the dataset is
    split on the attribute
    :param dataset: The dataset to analyse
    :param subsets: The data subsets to compare in order to compute the information gain
    :return: float
    """
    rows, cols = dataset.shape
    h = entropy(dataset)
    classes_with_na = dict()  # TODO get probabilities for each class for entries that have NaN value for this attr

    return h - sum([e(s, classes_with_na, rows) for s in subsets])


def information_value(dataset: np.ndarray, subsets):
    rows, cols = dataset.shape
    proportions = [float(subset.shape[0]) / rows for subset in subsets]
    return - sum([px * np.log2(px) for px in proportions])


def threshold_spilt(dataset: np.ndarray, attribute: int, threshold: Union[float, int]):
    na_values = dataset[dataset[:, attribute] == np.nan]
    others = dataset[dataset[:, attribute] != np.nan]
    return (
        others[others[:, attribute] <= threshold],
        others[others[:, attribute] > threshold]
    )


def gain_ratio(dataset: np.ndarray, subsets):
    return information_gain(dataset, subsets) / information_value(dataset, subsets)


def best_threshold(dataset: np.ndarray, attribute: int):
    # NOTE: we remove the last value as the attribute cannot be greater than this value
    values = sorted(set(dataset[:, attribute]))[:-1]
    ratios = [gain_ratio(dataset, threshold_spilt(dataset, attribute, threshold)) for threshold in values]
    maxed = max(ratios)
    return values[ratios.index(maxed)], maxed
