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
from sklearn.utils import class_weight

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
features = ['Customer Age (in months)', 'CHI Score Month 0', 'CHI Score 0-1', 'Support Cases Month 0','Support Cases 0-1', 'SP Month 0', 'SP 0-1', 'Logins 0-1','Blog Articles 0-1', 'Views 0-1', ' Days Since Last Login 0-1']

feature_matrix = raw_df[features]

#%% customer segmentation, group_a 0-6, b 7-12, c 13-18, d 18+. 

group_a = feature_matrix[feature_matrix['Customer Age (in months)']<=6]

churn_result_a = raw_df[raw_df['Customer Age (in months)']<=6]['Churn (1 = Yes, 0 = No)']

group_b = feature_matrix[(feature_matrix['Customer Age (in months)']>6) & (feature_matrix['Customer Age (in months)'] <=12) ]

churn_result_b = raw_df[(raw_df['Customer Age (in months)']>6) & (raw_df['Customer Age (in months)'] <=12)]['Churn (1 = Yes, 0 = No)']

group_c = feature_matrix[(feature_matrix['Customer Age (in months)']>12) & (feature_matrix['Customer Age (in months)'] <=18) ]

churn_result_c = raw_df[(raw_df['Customer Age (in months)']>12) & (raw_df['Customer Age (in months)'] <=18)]['Churn (1 = Yes, 0 = No)']

group_d = feature_matrix[feature_matrix['Customer Age (in months)']>18]
churn_result_d = raw_df[raw_df['Customer Age (in months)']>18]['Churn (1 = Yes, 0 = No)']


#%% loop through 4 groups and run 100 times for each decision tree
feature_set = [group_a.iloc[:, 1:], group_b.iloc[:, 1:], group_c.iloc[:, 1:], group_d.iloc[:, 1:]]

result_set = [churn_result_a, churn_result_b, churn_result_c, churn_result_d]

name_set = ['0-6', '7-12', '13-18', '18+']

feature_importance = pd.DataFrame()

features_name = [ 'CHI Score Month 0', 'CHI Score 0-1', 'Support Cases Month 0','Support Cases 0-1', 'SP Month 0', 'SP 0-1', 'Logins 0-1','Blog Articles 0-1', 'Views 0-1', ' Days Since Last Login 0-1']

for features, results, names in zip(feature_set, result_set, name_set):
    
    fea_importance = pd.DataFrame()
    for i in range (0, 1000):
    
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(results),
                                                 results)

        dt_cla = DecisionTreeClassifier(max_depth=None, max_features=None, max_leaf_nodes=None, class_weight ={0:class_weights[0], 1:class_weights[1]})

        dt_cla = dt_cla.fit(features, results)

#        fea_importance.join(
        fea_importance = fea_importance.append(pd.DataFrame(dt_cla.tree_.compute_feature_importances(normalize=False)).T)
        
    feature_importance[names] = fea_importance.mean(axis=0)

#%%
    
    
    
    
#%% plot decision tree

dot_data = StringIO()
out = export_graphviz(dt_cla, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                     feature_names=features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())

out = export_graphviz(dt_cla, out_file=r'C:\Users\s1883483\Desktop\2018 Rotman Datathon\output')