#!/home/ubuntu/anaconda2/envs/tensorflow/bin/python

import pickle
import argparse

parser = argparse.ArgumentParser(prog='print_pickle.py',
                                 description='show contents of specified pickle file',
                                 epilog="""example: 
./print_pickle.py best.pickle
"""
)

parser.add_argument('picklefile')
args = parser.parse_args()

picklefile = args.picklefile
const_spend_pct, var_spend_pcts, stock_allocations, bond_allocations = pickle.load( open(picklefile, "rb" ) )

print ("const_spend = %.12f" % const_spend_pct)
print ("var_spend_pcts = pd.Series(%s)" % str(list(var_spend_pcts)))
print ("stock_allocations = pd.Series(%s)\n" %str(list(stock_allocations)))

