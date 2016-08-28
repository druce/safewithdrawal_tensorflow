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

fileprefix = "best08"
bestfile = "%s.pickle" % (fileprefix)

max_unimproved_steps = 2000

gamma = 8.0

#Objective: 8315.064674

# const_spend = 2.321413
# var_spend_pcts = pd.Series([0.021015604501457775, 0.021761051829444631, 0.022312098346990435, 0.022785170076322969, 0.023285983064484993, 0.023897465220170052, 0.024584673876801872, 0.02556106756991109, 0.026657864441448173, 0.028031748201320435, 0.029551066581589736, 0.031201618742953394, 0.032978432086452118, 0.034516254916809298, 0.036027857701909138, 0.037763940480250287, 0.03992129323858909, 0.042635694985269881, 0.045638329119485004, 0.049069352739346678, 0.052383268763417638, 0.056951126091794861, 0.063470193195596478, 0.070974811737827201, 0.082180160879307573, 0.098169174319082841, 0.1205906552280696, 0.15769373320000857, 0.23376809386762137, 0.51005368542831198])
# stock_allocations = pd.Series([0.82085705309182722, 0.8208564375532369, 0.80809230790394848, 0.80474242187125467, 0.80321803760810162, 0.80214299804721623, 0.80178790048600157, 0.7839705620587375, 0.77739050153152156, 0.77699016168709201, 0.77517208520407443, 0.76706047015389667, 0.76676220145412832, 0.76576837231963391, 0.76098570290996814, 0.74113354059879621, 0.73793102049167558, 0.73650905089885166, 0.72707794679494286, 0.72393066589418387, 0.7210099158662584, 0.71370848573117784, 0.7038219623712294, 0.68848317679023907, 0.61956979054659567, 0.61331107236876559, 0.59738860596743892, 0.59391944015033249, 0.59164222259062249, 0.53441829378265526])

# startval = 100
# years_retired = 30
# const_spend_pct = .02
# const_spend = startval * const_spend_pct

# var_spend_pcts = pd.Series(np.ones(years_retired) * 0.02)
# var_spend_pcts[-1]=1.0
# stock_allocations = pd.Series(np.ones(years_retired) * 0.65)

startval = 100
years_retired = 30
# 1.5% constant spending
const_spend_pct = 0.015
const_spend = startval * const_spend_pct

# var spending a function of years left
var_spend_pcts = pd.Series([ 1.0 / (years_retired - ix) - 0.01 for ix in range(years_retired)])

# 50% stocks + 1% * years remaining
stock_allocations = pd.Series([0.50 + 0.01 * (years_retired - ix)  for ix in range(years_retired)])
bond_allocations = 1 - stock_allocations

#Objective: 4.576467

const_spend = 1.486143
var_spend_pcts = pd.Series([0.02926346176255188, 0.030964094008338886, 0.032032860931716992, 0.032511355803879725, 0.033521199917626802, 0.03508962731999158, 0.036280953234574864, 0.037776153620537371, 0.040117191398331364, 0.04235504955888323, 0.044531955751014672, 0.047020364984040948, 0.049954966423005148, 0.052296790706288586, 0.054440339496671153, 0.057598502174747303, 0.060451827443809933, 0.063598790912842151, 0.067378844701121612, 0.071531499750243893, 0.074462730671590663, 0.079736458902363611, 0.088033534063302249, 0.099820298182881739, 0.12082514029592897, 0.15315150052956999, 0.20309429469028545, 0.28629981434729418, 0.44992452769434604, 0.98999999999999999])
stock_allocations = pd.Series([0.80309601271942199, 0.79347452459017465, 0.78385303646092719, 0.77423154833167984, 0.76461006020243238, 0.75498857207318504, 0.74536708394393769, 0.73574559581469023, 0.72612410768544289, 0.71650261955619543, 0.70688113142694808, 0.69725964329770074, 0.68763815516845328, 0.67801666703920593, 0.66839517890995848, 0.65877369078071113, 0.64915220265146378, 0.63953071452221633, 0.62990922639296898, 0.62028773826372152, 0.61066625013447418, 0.60104476200522683, 0.59142327387597937, 0.58180178574673203, 0.57218029761748457, 0.56255880948823722, 0.55293732135898988, 0.54331583322974242, 0.53369434510049496, 0.52407285697124761])

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

