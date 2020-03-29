"""
The aim of this fi:e is to have a generic base for any decision tree algorithm
to avoid code duplication as much as possible
"""
from typing import Generic, TypeVar
from networkx import DiGraph
from pandas import Series, DataFrame
import numpy as np


class GenericNode:
    def __init__(self, name, edges):
        self.name = name
        """Name of the attribute"""
        self.edges = edges
        """Attribute values"""
        if edges is not None:
            self.child = dict.fromkeys(edges)
        else:
            self.child = None
        """Children to access when following edges"""

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class GenericClass(GenericNode):
    def __init__(self, name, value):
        super().__init__(name, None)
        self.value = value

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


Node = TypeVar('Node', bound=GenericNode)


class GenericModel(Generic[Node]):
    def __init__(self, tree: DiGraph, root: Node):
        self.tree = tree
        self.root = root

    def decode_prediction(self, attributes: Series):
        return self._browse_tree(self.root, attributes)

    def _browse_tree(self, node: Node, attributes: Series):
        raise NotImplementedError()

    def confusion_matrix(self, data_set: DataFrame):
        classes = list(set(data_set.iloc[:, -1].values))
        class_count = len(classes)
        matrix = DataFrame(
            np.zeros((class_count, class_count), dtype=np.uint),
            columns=classes,
            index=classes
        )

        for index, row in data_set.iterrows():
            prediction = self.decode_prediction(row)
            matrix.at[row.get(-1), prediction] += 1

        return matrix
