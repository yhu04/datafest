__author__ = 'zhangyin'


import os
import pandas as pd
import numpy as np
import math



user_id_order = pd.read_csv("/Users/zhangyin/python projects/2016DataFestUVa/NN/examples/user_id_order.csv").ix[:,1]

user_label = pd.read_csv("/Users/zhangyin/python projects/2016DataFestUVa/NN/examples/user_label.csv").ix[:,1:]

UserBehaviorX = pd.read_csv("/Users/zhangyin/python projects/2016DataFestUVa/NN/examples/UserBehaviorX.csv").ix[:,1:]

UserBehavior_comp_id_date = pd.read_csv("/Users/zhangyin/python projects/2016DataFestUVa/UserBehavior_comp_id_date.csv",header=None)

fullvisitorid = UserBehavior_comp_id_date.ix[:,2]


# merge UserBehaviorX and user_label
new = UserBehaviorX.merge(user_label,left_index=True, right_index=True,how ='inner')

new.to_csv("DataForNN.csv"，header=None,rownames=False)

