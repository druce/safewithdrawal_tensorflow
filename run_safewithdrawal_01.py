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

gamma = 1.0
fileprefix = "best%02d" % gamma
bestfile = "%s.pickle" % (fileprefix)

max_unimproved_steps = 2000

# startval = 100
# years_retired = 30
# const_spend_pct = .02
# const_spend = startval * const_spend_pct

# var_spend_pcts = pd.Series(np.ones(years_retired) * 0.02)
# var_spend_pcts[-1]=1.0
# stock_allocations = pd.Series(np.ones(years_retired) * 0.65)

startval = 100
years_retired = 30

#Objective: 27.482298

const_spend = 0.000000
var_spend_pcts = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
stock_allocations = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

#Objective: 9.3197835039548

const_spend = 0.031171

var_spend_pcts = pd.Series([0.033031715082124893, 0.034179658923378056, 0.035413235673810203, 0.03674041861668869, 0.038170522767858685, 0.039714318485430097, 0.041384780016445105, 0.043198654605266659, 0.045174497287581369, 0.047337726241673178, 0.049718885196671697, 0.052349339898335248, 0.05527231487699432, 0.058542485334810268, 0.062223822202370035, 0.066393675531524268, 0.071154938924494274, 0.076645922264574853, 0.083051431231744757, 0.090624657670722322, 0.099713290522915235, 0.11081878154864604, 0.12469530025942732, 0.14253675149889414, 0.16632259898937693, 0.19961900493553258, 0.24956175256190519, 0.33278703965562378, 0.49873317722521254, 1.000000000])
stock_allocations = pd.Series([0.99997992186395857, 0.99997829495822088, 0.99997772801721752, 0.99997739871117364, 0.999981881805168, 0.99997168753575605, 0.99998297109656353, 0.99997718621226794, 0.99995206407097303, 0.99993387591616201, 0.99991562318746663, 0.99989638853120155, 0.99987202763403471, 0.99986392544055869, 0.99985827635323554, 0.99985583354131069, 0.99981745777551367, 0.99980538100009697, 0.99977354919295003, 0.99968776769934442, 0.99963126320848172, 0.99956977316432905, 0.99943095802616133, 0.99939580691208751, 0.99922584340950216, 0.98242971040593718, 0.97558375257576069, 0.95743881935199782, 0.9327435399245898, 0.91540890472708358])

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

