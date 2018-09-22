# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:31:28 2018
Team : AML Analytics & Reporting
Author: Chandler Qian
Content : 
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from sklearn import preprocessing

#%% data exploration for existing parameters

raw_df = pd.read_excel(r"C:\Users\s1883483\Desktop\2018 Rotman Datathon\Churn+Case+Data+UVAQA0806X.xlsx", sheetname="Case Data")

#%% chanllenges 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

#%% build decision tree
features = ['ID', 'Customer Age (in months)', 'Churn (1 = Yes, 0 = No)',
       'CHI Score Month 0', 'CHI Score 0-1', 'Support Cases Month 0',
       'Support Cases 0-1', 'SP Month 0', 'SP 0-1', 'Logins 0-1',
       'Blog Articles 0-1', 'Views 0-1', ' Days Since Last Login 0-1']

feature_matrix = raw_df[features]

churn_result = raw_df['Churn (1 = Yes, 0 = No)']

dt_cla = DecisionTreeClassifier()

dt_cla = dt_cla.fit(feature_matrix, churn_result)

#%% plot decision tree

dot_data = StringIO()
export_graphviz(dt_cla, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
    
    