from os.path import dirname

import pandas as pd
from matplotlib import pyplot
from networkx import draw

from C4_5 import generate_c4_5_tree
from id3 import generate_id3_tree

# ======================
#          ID3
# =====================

# Load csv
app = pd.read_csv(dirname(__file__) + '/../data/golf.csv', na_values='?')
# Compute model
model = generate_id3_tree(app)


draw(model.tree, with_labels=True, arrows=True)
pyplot.show()


## Confusion matrices
print('Matrice de confusion en apprentissage :')
print(model.confusion_matrix(app))

# no dataset to use as prediction
# print('Matrice de confusion en prédiction :')
# print(model.confusion_matrix(pred))

# ======================
#          C4.5
# =====================

# Load csv
app = pd.read_csv(dirname(__file__) + '/../data/golf.csv', na_values='?')
# Compute model
model = generate_c4_5_tree(app)


draw(model.tree, with_labels=True, arrows=True)
pyplot.show() # cannot find (after a lot of research) how to show edges value accurately


## Confusion matrices
print('Matrice de confusion en apprentissage :')
print(model.confusion_matrix(app))

# no dataset to use as prediction
# print('Matrice de confusion en prédiction :')
# print(model.confusion_matrix(pred))

