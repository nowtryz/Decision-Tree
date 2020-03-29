"""
Data are taken from the Quinlan's article available at the root of this project
"""
from os.path import dirname
from unittest import TestCase

from pandas import read_csv

from id3 import entropy, information_gain
from utils import pd_index

app = read_csv(dirname(__file__) + '/../data/golf.csv')
dataset = app.to_numpy()


class Test(TestCase):
    def test_entropy(self):
        self.assertAlmostEqual(entropy(dataset), 0.940, places=3)

    def test_temperature_gain(self):
        self.assertAlmostEqual(information_gain(dataset, pd_index(app, "temp")), 0.029, places=2)

    def test_humidity_gain(self):
        self.assertAlmostEqual(information_gain(dataset, pd_index(app, "humidity")), 0.151, places=2)

    def test_wind_gain(self):
        self.assertAlmostEqual(information_gain(dataset, pd_index(app, "wind")), 0.048, places=2)
