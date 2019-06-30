#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time-stamp: <2019-02-12 19:43:14 ycopin>

"""
CTIO slitless spectrum analysis.
"""

from __future__ import division, print_function

import numpy as N
import xarray as X
import pandas as D
import matplotlib.pyplot as P

__author__ = "Yannick Copin <y.copin@ipnl.in2p3.fr>"
__version__ = "Tue Feb 12 18:59:26 2019"


def plot(data, xlabel, ylabel, ax=None):

    if ax is None:
        fig, ax = P.subplots(1, 1, 1)

    l, = ax.plot(xlabel, ylabel, data=data, label=ylabel)

    if not ax.xaxis.get_label().get_text():
        ax.set_xlabel(xlabel)
    if not ax.yaxis.get_label().get_text():
        ax.set_ylabel(ylabel)

    return l


if __name__ == '__main__':

    csvname = "reduc_20170530_130_table.csv"

    data = X.Dataset.from_dataframe(D.read_csv(csvname))
    data.info()

    # Dispersion solution
