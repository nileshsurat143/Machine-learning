# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:06:52 2020

@author: nil
"""


#pip install mlxtend 
#pip install --no-binary :all: mlxtend




import numpy as np
import pandas as pd

transactions = [[1, 2, 5],
                [2, 4],
                [2, 3],
                [1, 2, 4],
                [1, 3],
                [2, 3],
                [1, 3],
                [1, 2, 3, 5],
                [1, 2, 3]]

'''
# Importing dataset
dataset = pd.read_csv('Market.csv',names=np.arange(1,21))

# Preprocesing data to fit model
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist)
  
'''



from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_x = pd.DataFrame(te_ary, columns=te.columns_) # encode to onehot

# Train model using Apiori algorithm 
# ref = https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x, min_support=0.3, use_colnames=True)

df_rules = association_rules(df_setd, metric = 'support', min_thresold = 0.3, support_only = True)
 
# if you use only "support", it called "ECLAT"

