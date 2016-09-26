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

gamma = 4.0

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
const_spend = startval * const_spend_pct

# var spending a function of years left
var_spend_pcts = pd.Series([ 0.5/(30-ix) for ix in range(30)])
var_spend_pcts[29] = 1.0

# stocks starting at 82%, decreasing 0.5% per year
stock_allocations = pd.Series([0.82 - 0.005* ix for ix in range(30)])

#Objective:  5.653304

const_spend = 0.013135043394

var_spend_pcts = pd.Series([0.035822978028495614, 0.037142880037028099, 0.038232214787301988, 0.039230415975672818, 0.040201921370750057, 0.04128306632974512, 0.042562738008764228, 0.044203187483426658, 0.04612997210464486, 0.048403252082914246, 0.050913960441316269, 0.053804872403093118, 0.056961354837341847, 0.059937273709488398, 0.062775232619198637, 0.066025078674551443, 0.070108404481856915, 0.075376643015682129, 0.081397919560765669, 0.087812983760457097, 0.094573649661557693, 0.10292659618285707, 0.11353980019663899, 0.12569730816023159, 0.14198879320676885, 0.16414995001568367, 0.19648563495632851, 0.25250659931593217, 0.3706052129208936, 1.0])
stock_allocations = pd.Series([0.90099032199651863, 0.89924731857030904, 0.89924731828348214, 0.89876830969341182, 0.89770857617754174, 0.89503554351652825, 0.89232180232291247, 0.88505129512940295, 0.8648266951006971, 0.86372639925213479, 0.85995907966929552, 0.85302430311396027, 0.85289466074670783, 0.85270233833392206, 0.85157005570710653, 0.84601302356094343, 0.83930541580936513, 0.83354308642897557, 0.82595372913293841, 0.82174793960063786, 0.80903946044432207, 0.80583306771707164, 0.79764354604204957, 0.79580225813480721, 0.75787634532529125, 0.75092813511924261, 0.73890315821297692, 0.73198221026156862, 0.72778224605194297, 0.69545344502127571])

bond_allocations = 1 - stock_allocations

# save starting scenario
pickle_list = [const_spend, var_spend_pcts, stock_allocations, bond_allocations]
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

