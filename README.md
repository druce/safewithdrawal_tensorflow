# safewithdrawal_tensorflow

# Safe Withdrawal with Certainty Equivalent Cash Flow and TensorFlow

run for gamma = 8 with 

`./run_safewithdrawal_08.py`

  * sets up variables to be optimized 
  * saves in a pickle file
  * repeatedly calls safewithdrawal.py
  * reduces learning rate each time

`./safewithdrawal.py` learning_rate steps gamma fileprefix

  * loads variables from `fileprefix`.pickle
  * runs optimization using `gamma` and `learning_rate` until no improvement for specified `steps`
  * saves variables and csv files summarizing outcome
  
Safe Withdrawal with Certainty Equivalent Spending and Tensorflow Aug 2016.ipynb
  * Jupyter notebook which allows you to run step by step, includes comments and graphs
  * However running for a few hours in Jupyter not recommended, browser or Jupyter tends to hang
  
Please contact with any comments, pull requests 
  * Run efficiently on GPU
  * Use better optimizer like AdamOptimizer instead of GradientDescentOptimizer
    * adaptive learning rate (vs repeated calls with lower learning rate)
    * momentum so less likely to get stuck.




