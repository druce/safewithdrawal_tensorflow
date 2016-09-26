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

const_spend_pct = 0.000210991438
var_spend_pcts = pd.Series([0.033091538348404496, 0.034238168285506382, 0.035469702440122583, 0.036794378152860914, 0.038221756778048857, 0.03976288047503998, 0.041430947662087859, 0.043242686669740876, 0.045216839687742649, 0.047378561748440171, 0.04975812222825144, 0.052387429460561782, 0.055309651974463532, 0.058579229736857016, 0.062260304236144169, 0.066431027576827867, 0.071195216736901532, 0.076691471776952069, 0.083104783014253336, 0.090686442187799754, 0.099783243431139951, 0.11089638349654848, 0.12478101422016996, 0.14262272988447125, 0.16639735490582469, 0.19966642400185314, 0.24955549906491153, 0.33270239606716523, 0.49854063104140045, 1.0])
stock_allocations = pd.Series([0.99999999932534844, 0.99997650919314807, 0.99995061726681489, 0.99992473029357398, 0.99933647980151297, 0.99915824051192981, 0.99801222176142057, 0.9979130315525, 0.99767125200215423, 0.99762089448679248, 0.99748512848933513, 0.99744284142382889, 0.99734447386668101, 0.99709686697171851, 0.99700226658406843, 0.99670993806194219, 0.99638045411265985, 0.99628135924967531, 0.99610184313223671, 0.99588814803982129, 0.99574876269478818, 0.99557475808121887, 0.99508658081662371, 0.99501534744474973, 0.99468563684794653, 0.98122175151927205, 0.97403705686938635, 0.9563148825731429, 0.93175496886826248, 0.91435187362375314])

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

