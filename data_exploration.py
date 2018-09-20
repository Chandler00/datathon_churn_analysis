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

class data_exp:
    
    def __init__(self,):
        
    
    def plot_corr(self):
        stock_df = normalize_stock(self.stock_df)
        stock_df = date_range(self.date_start, self.date_end, stock_df)
        corr = stock_df.T.corr()
        corr_top_index = corr[self.tar_equity].sort_values(ascending=False)[:self.top_k].index.tolist()
        corr_top = stock_df.T[corr_top_index ].corr()
        plt.figure(figsize=(16, 16))
        corr_map = sns.heatmap(corr_top, xticklabels=corr_top.columns, yticklabels=corr_top.columns, annot=True)
        figure = corr_map.get_figure()    
        figure.savefig(r'C:\Users\s1883483\Desktop\Advanced analytics projects\coding cafe\KNN output\stock_corr.png', dpi=800)