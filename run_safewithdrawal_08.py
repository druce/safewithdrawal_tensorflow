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

import tensorflow as tf
from safewithdrawal import *

gamma = 8.0

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

const_spend_pct = 0.018033736915
var_spend_pcts = pd.Series([0.027420789200650691, 0.028952687799914788, 0.029987640366612587, 0.030563608815231274, 0.031468009954100477, 0.0327892165458714, 0.033812194391987072, 0.035195378137199251, 0.037327252304951188, 0.039532656754857148, 0.041711339480165768, 0.044205216561892961, 0.047114695564658048, 0.049375968207551808, 0.051345728157015109, 0.05421706340461499, 0.057174919670928206, 0.060803826336315064, 0.065058176584335367, 0.069765865219855477, 0.073504852805054741, 0.079201426822341936, 0.087219117566384868, 0.095745358220279422, 0.10919882479478223, 0.12781818289102989, 0.15488310518171028, 0.20138642112957703, 0.29915169759218702, 1.0])
stock_allocations = pd.Series([0.85995594522555152, 0.85919296367537157, 0.85473424640534235, 0.85205353144872398, 0.84754303090697891, 0.84731613048078414, 0.84674123511449917, 0.83207329884146453, 0.81886882987838872, 0.81886882879077094, 0.81787428397721995, 0.81082294011238876, 0.80935846467721007, 0.80915247713021987, 0.80869082405588244, 0.79235039525216477, 0.78697518924653442, 0.78566021347854842, 0.77606352982164872, 0.77226108408967586, 0.76418701504255893, 0.75957408240400748, 0.7506635662237392, 0.74210994715180123, 0.68810212447786767, 0.68212008875931995, 0.66794616058167322, 0.66270963157142881, 0.65979617905489518, 0.61499306500466799])

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



