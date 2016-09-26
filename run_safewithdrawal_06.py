#!/home/ubuntu/anaconda2/bin/python

# MIT License

# Copyright (c) 2016 Druce Vertes drucev@gmail.com

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import argparse
import pickle
from time import strftime
import sys
import os

import numpy as np
import pandas as pd

gamma = 6.0

fileprefix = "best%02.0f" % gamma
bestfile = "%s.pickle" % (fileprefix)

max_unimproved_steps = 1000

# startval = 100
# years_retired = 30
# const_spend_pct = .02
# const_spend = startval * const_spend_pct

# var_spend_pcts = pd.Series(np.ones(years_retired) * 0.02)
# var_spend_pcts[-1]=1.0
# stock_allocations = pd.Series(np.ones(years_retired) * 0.65)

startval = 100
years_retired = 30

const_spend_pct = .0225

# var spending a function of years left
var_spend_pcts = pd.Series([ 0.5/(30-ix) for ix in range(30)])
var_spend_pcts[29] = 1.0

# stocks starting at 82%, decreasing 0.5% per year
stock_allocations = pd.Series([0.82 - 0.005* ix for ix in range(30)])

#Objective: 5.056278

const_spend_pct = 0.015637933571
var_spend_pcts = pd.Series([0.031606123286676821, 0.033096808496424615, 0.034176874786077449, 0.034949059621799367, 0.035890789552150887, 0.037116351576046536, 0.038256812396706585, 0.039784074922038948, 0.041850514856008901, 0.044113104719702513, 0.046479531291527841, 0.049188553520785189, 0.052237088536900282, 0.054826487679491116, 0.05719659838516862, 0.060236417433033075, 0.06372908174792162, 0.068170731957061817, 0.073291375378079365, 0.078832696364871821, 0.084046153747339689, 0.091046547868110006, 0.10034482939483094, 0.11066289255761518, 0.1255268715113966, 0.14590561544900946, 0.17559280570819452, 0.22684587500024678, 0.33478357325498115, 1.0])
stock_allocations = pd.Series([0.88033446277449279, 0.87918369248956052, 0.87699906807696382, 0.87541489848777843, 0.87254543385561067, 0.87112127695141239, 0.8696038278716699, 0.85852400031522713, 0.84145611984667457, 0.84145611982772339, 0.83898260822198845, 0.83191026515233402, 0.83112108486165093, 0.83100258791320347, 0.83025217396697193, 0.81919450523261073, 0.81311531977557161, 0.809624259366025, 0.80101985702945411, 0.79700856489108074, 0.78660821862681019, 0.7827212663418508, 0.77416817507231639, 0.76897655138636212, 0.72297797819724985, 0.7165312149934886, 0.70342618929327727, 0.69734141791783077, 0.69379260335071868, 0.65522743833716124])

bond_allocations = 1 - stock_allocations

# save starting scenario
pickle_list = [const_spend_pct, var_spend_pcts, stock_allocations, bond_allocations]
pickle.dump( pickle_list, open( bestfile, "wb" ) )


# start with a learning rate that learns quickly, gradually reduce it
# run once with 50 or 100 steps to see which learning rates are effective
# then plug in that solution and run each til no improvement for a large number of steps

for learning_rate in [
        #0.00001, # too coarse, may be NaN
        0.00003, # too coarse, may be NaN
        0.000001, # coarse
        0.000003, # coarse
        0.0000001, # workhorse
        0.00000003, 
        0.00000001, # diminishing returns
        0.000000003,
        #0.000000001, #superfine
        #0.0000000003, 
        #0.0000000001, 
        #0.00000000001, 
]:
    cmdstr = './safewithdrawal.py %.12f %d %f %s' % (learning_rate, max_unimproved_steps, gamma, fileprefix)
    print(cmdstr)
    os.system(cmdstr)

