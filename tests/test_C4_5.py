"""
Data used below to test the algorithm have been calculated by hand using
the "golf_bis.csv" data to have an idea of the results to expect.
"""
from os.path import dirname
from unittest import TestCase

from pandas import read_csv

from utils import pd_index
from C4_5 import *


app = read_csv(dirname(__file__) + '/../data/golf_bis.csv')
dataset = app.to_numpy()


class TestC45(TestCase):
    def util_threshold_checks(self, attribute, threshold,
                              set_entropy, below_entropy, above_entropy,
                              gain, _split_info, ratio):

        subsets = threshold_spilt(dataset, attribute, threshold)
        less_or_equal, greater = subsets

        self.assertAlmostEqual(
            entropy(dataset), set_entropy,
            places=3, msg="incorrect dataset entropy"
        )
        self.assertAlmostEqual(
            entropy(less_or_equal), below_entropy,
            places=3, msg="incorrect below threshold entropy"
        )
        self.assertAlmostEqual(
            entropy(greater), above_entropy,
            places=3, msg="incorrect above threshold entropy"
        )
        self.assertAlmostEqual(
            information_gain(dataset, subsets), gain,
            places=3, msg="incorrect information gain"
        )
        self.assertAlmostEqual(
            information_value(dataset, subsets), _split_info,
            places=3, msg="incorrect split info"
        )
        self.assertAlmostEqual(
            gain_ratio(dataset, subsets), ratio,
            places=2, msg="incorrect gain ratio"
        )

    def test_non_discreet_humidity_65(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=65,
                                   set_entropy=0.940,
                                   below_entropy=.0,
                                   above_entropy=0.961,
                                   gain=0.048,
                                   _split_info=0.371,
                                   ratio=0.13)

    def test_non_discreet_humidity_70(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=70,
                                   set_entropy=0.940,
                                   below_entropy=0.811,
                                   above_entropy=0.971,
                                   gain=0.015,
                                   _split_info=0.863,
                                   ratio=0.02)

    def test_non_discreet_humidity_80(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=80,
                                   set_entropy=0.940,
                                   below_entropy=0.592,
                                   above_entropy=0.985,
                                   gain=0.152,
                                   _split_info=1,
                                   ratio=0.15)

    def test_non_discreet_humidity_85(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=85,
                                   set_entropy=0.940,
                                   below_entropy=0.811,
                                   above_entropy=1.0,
                                   gain=0.048,
                                   _split_info=0.985,
                                   ratio=0.048)

    def test_non_discreet_humidity_86(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=86,
                                   set_entropy=0.940,
                                   below_entropy=0.764,
                                   above_entropy=0.971,
                                   gain=0.102,
                                   _split_info=0.940,
                                   ratio=0.108)

    def test_non_discreet_humidity_90(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=90,
                                   set_entropy=0.940,
                                   below_entropy=0.845,
                                   above_entropy=0.918,
                                   gain=0.079,
                                   _split_info=0.750,
                                   ratio=0.105)

    def test_non_discreet_humidity_91(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=91,
                                   set_entropy=0.940,
                                   below_entropy=0.918,
                                   above_entropy=1.0,
                                   gain=0.010,
                                   _split_info=0.592,
                                   ratio=0.017)

    def test_non_discreet_humidity_95(self):
        self.util_threshold_checks(attribute=pd_index(app, "humidity"),
                                   threshold=95,
                                   set_entropy=0.940,
                                   below_entropy=0.961,
                                   above_entropy=.0,
                                   gain=0.0474,
                                   _split_info=0.371,
                                   ratio=0.128)

    def test_best_threshold_humidity(self):
        humidity, maxed = best_threshold(dataset, pd_index(app, "humidity"))
        self.assertEqual(
            80, humidity,
            msg="Best threshold for humidity should be 80"
        )
