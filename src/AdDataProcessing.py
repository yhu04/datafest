
import os
import pandas as pd
import numpy as np
import math

os.chdir("/Users/zhangyin/python projects/2016DataFestUVa/")

# class adwords_load_process_save:

def for_var4(v):
    if (v==' --'):
        return np.nan
    elif (str(v).startswith("auto:")):
        return str(v)[6:]
    else:
        return v

def currency2num(x):  # change "1,123.01" -> 1123.01
    x = str(x)
    return float(x.replace(",",""))

def p2f(x):  # change "95.0%" -> 0.95
    return float(x.strip('%'))/100

def ZeroOneScale(var):
    var = np.array(var,dtype=float)
    var = (var - np.nanmin(var))/ (np.nanmax(var)-np.nanmin(var))
    return var.reshape((var.shape[0],1))

def load_process_save(out_file):
    print "loading"
    ad_data = pd.read_csv("./DFData/approved_adwords_v3.csv")

    a = np.array([4,5])
    b = np.arange(7,28)
    id = np.hstack((a,b))
    selected_ad_data = ad_data.ix[:,id]

    # remove missing values, we find that the missing values only exist in var4
    var4 = selected_ad_data.ix[:,4]
    var4 = np.array([float(for_var4(v)) for v in var4])
    var4 = ZeroOneScale(var4)
    np.unique(var4)
    rows_id = np.where(np.logical_not(np.isnan(var4)))[0] #rows_id are ids of not massing values

    selected_ad_data = ad_data.ix[rows_id,id]

    ##### process data #####
    print("data processing")
    names = selected_ad_data.keys()
    ####

    camp_id = ad_data.ix[rows_id,0]
    week = ad_data.ix[rows_id,5]

    np.unique(selected_ad_data.ix[:,0])
    var0 = pd.get_dummies(selected_ad_data.ix[:,0])
    var0 = var0.values
    np.unique(var0)

    # var1 = pd.get_dummies(selected_ad_data.ix[:,1])
    # var1.shape
    # np.unique(selected_ad_data.ix[:,1])

    var2 = pd.get_dummies(selected_ad_data.ix[:,2])
    var2 = var2.values
    var2.shape
    np.unique(selected_ad_data.ix[:,2])

    var3 = pd.get_dummies(selected_ad_data.ix[:,3])
    var3 = var3.values
    var3.shape
    np.unique(selected_ad_data.ix[:,3])

    var4 = selected_ad_data.ix[:,4]
    var4 = np.array([float(for_var4(v)) for v in var4])
    var4 = ZeroOneScale(var4)
    np.unique(var4)

    var5 = selected_ad_data.ix[:,5]
    var5 = ZeroOneScale(var5)
    np.unique(var5)

    var6 = ZeroOneScale(selected_ad_data.ix[:,6])
    np.unique(var6)

    var7 = selected_ad_data.ix[:,7]
    var7 = ZeroOneScale([p2f(v) for v in var7])
    np.unique(var7)

    var8 = ZeroOneScale(selected_ad_data.ix[:,8])
    np.unique(var8)

    var9 = ZeroOneScale([currency2num(v) for v in selected_ad_data.ix[:,9]])
    np.unique(var9)

    var10 = ZeroOneScale([currency2num(v) for v in selected_ad_data.ix[:,10]])
    np.unique(var10)

    var11 = ZeroOneScale(selected_ad_data.ix[:,11])
    np.unique(var11)

    var12 = ZeroOneScale(selected_ad_data.ix[:,12])
    np.unique(var12)
    var12.shape

    var13 = ZeroOneScale(selected_ad_data.ix[:,13])
    np.unique(var13)

    var14 = ZeroOneScale(selected_ad_data.ix[:,14])
    np.unique(var14)

    var15 = ZeroOneScale(selected_ad_data.ix[:,15])
    np.unique(var15)

    var16 = ZeroOneScale([p2f(v) for v in selected_ad_data.ix[:,16]])
    np.unique(var16)

    var17 = ZeroOneScale(np.array([currency2num(v) for v in selected_ad_data.ix[:,17]]))
    np.unique(var17)

    var18 = ZeroOneScale(np.array([currency2num(v) for v in selected_ad_data.ix[:,18]]))
    np.unique(var18)

    var19 = ZeroOneScale(np.array([currency2num(v) for v in selected_ad_data.ix[:,19]]))
    np.unique(var19)

    var20 = ZeroOneScale(np.array([currency2num(v) for v in selected_ad_data.ix[:,20]]))
    np.unique(var20)

    var21 = ZeroOneScale(np.array([currency2num(v) for v in selected_ad_data.ix[:,21]]))
    np.unique(var21)
    var21.shape

    var22 = ZeroOneScale(np.array([p2f(v) for v in selected_ad_data.ix[:,22]]))
    np.unique(var22)
    var22.shape

    print ("concatenating")
    df = np.hstack((var0,var2,var3, var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18,var19,var20,var21,var22))

    print ("writing to csv...")
    np.savetxt(out_file, df, delimiter=",")
    np.savetxt("camp_id.csv", camp_id.values, delimiter=",")

    camp_date = pd.concat([camp_id,week],axis=1)
    camp_date.to_csv("camp_id_date.csv",index=False,header=False)

    print "Done!"

if __name__ == "__main__":
    load_process_save(out_file="out.csv")




