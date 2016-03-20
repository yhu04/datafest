from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.isinteractive()
plt.interactive(False)
os.chdir("/Users/zhangyin/python projects/2016DataFestUVa/")


def prop_of_nan(x): # can be used to calculate the prop of nan in each field
    x = np.array(x)
    count = 0
    nr = x.shape[0]
    for i in range(nr):
        try:
            if np.isnan(x[i]):
                count += 1
        except:
            pass
    return count*1.0/nr

# read data
ub_data = pd.read_csv("./DFData/approved_ga_data_v2.csv")
nr,nc = ub_data.shape   # number of rows and cols of orginial dataset
names = ub_data.keys()  # names records the names of fields in the dataset
camp_id = ub_data['campaign_id'].values # extract the camp_id, which is the response of our classification problem

# Data manipulation 1: remove the records with missing camp_id
row_id = []
for i in range(nr):
    try:
        if np.isnan(camp_id[i]):
            pass
    except:
        row_id.append(i)
row_id = np.array(row_id)
ub_data = ub_data.ix[row_id,:]
nr,nc = ub_data.shape  # number of rows and cols of this dataset

date = ub_data["date"] # extract date
fullvisitorid = ub_data['fullvisitorid']
visitid = ub_data['visitid']
customer_id = ub_data['customer_id']

# Data manipulation 2: keep the variables we will take into consideration for classification
names_id =np.array([4,8,9,10,11,12,14,16,17,20,21,23,25,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])-1
names[names_id] # the names

# look at the prop of missing value in each variable in ub_data_2
ub_data_value = ub_data.values
for j in range(nc):
    if prop_of_nan(ub_data_value[:,j])>0:
        print (j,prop_of_nan(ub_data_value[:,j]))


# miss value proportion in each variable in ub_data_2
# 9 0.000407540184153
# 10 0.0585130999993
# 11 0.950169578161
# 17 1.0
# 27 0.524877219885
# 39 0.821069136775
# 40 0.790593419953
names[np.array([9,10,11,17,27,39,40])] # look at these variables with missing values
# Index([u'totals_pageviews', u'totals_timeonsite', u'totals_bounces',
#        u'referralpath', u'device_mobiledevicebranding', u'hits_isentrance',
#        u'hits_isexit'],
#       dtype='object')


# we remove 11,17,27,39,40th varialbes. Further, we use geonetwork_region as geo information, so delete col 33, 34, 36
final_name_id = np.setdiff1d(names_id,np.array([11,17,27,33, 34, 36, 39,40]))
final_name = names[final_name_id]

# remove the obs with nan in line9 and line10
line9 = np.array(ub_data_value[:,9],dtype=float)
line10 = np.array(ub_data_value[:,10],dtype=float)
not_nan_id = np.where(np.logical_and(~np.isnan(line9),~np.isnan(line10)))[0]
ub_data_clean = ub_data_value[:,final_name_id]
ub_data_clean = ub_data_clean[not_nan_id]
ub_data_clean.shape
final_name
len(final_name)

date = date.iloc[not_nan_id] #update date
fullvisitorid = fullvisitorid.iloc[not_nan_id]
visitid = visitid.iloc[not_nan_id]
customer_id = customer_id.iloc[not_nan_id]
# Data manipulation 3: convert categorical data to dummy variables and do normalization to every variable
# ub_data_clean = pd.DataFrame(ub_data_clean)
nr,nc = ub_data_clean.shape

## look at unique values of each variable
for i in range(nc):
    print (i,final_name[i])
    # print i,np.unique(ub_data_clean[:,i])

def print_cum_prop(ser):
    sum=0
    cum_ser = pd.Series.value_counts(ser)
    p = cum_ser/np.sum(cum_ser)
    for i in range(len(p)):
        sum += p[i]
        print (i,sum)

def ZeroOneScale(ser):
    var = np.array(ser,dtype=float)
    var = (var - np.nanmin(var))/ (np.nanmax(var)-np.nanmin(var))
    return pd.Series(var)

def KeepFirstk_Rest2Other(ser,k,other_name):
    ser_count = pd.Series.value_counts(ser)
    first_k_names = ser_count.keys()[:k]
    for i in range(len(ser)):
        if ser[i] not in first_k_names:
            ser[i] = other_name
    return ser

# num: 0,2,3,4,18,22
# cat: 10,11,12,14,15,17,19,23,24
# resp: 5

# 0. visitnumber: num
pd0 = pd.Series(ub_data_clean[:,0])
pd.Series.value_counts(pd0)

# 1. all total visit = 1, so remove this variable
pd1 = pd.Series(ub_data_clean[:,1])
pd.Series.value_counts(pd1)

# 2. totals_hits: num
pd2 = pd.Series(ub_data_clean[:,2])
pd.Series(ub_data_clean[:,2]).value_counts(dropna=False)

# 3. totals_pageviews: num
pd3 = pd.Series(ub_data_clean[:,3])
pd.Series.value_counts(pd.Series(ub_data_clean[:,3]))
# pd.Series(ub_data_clean[:,3]).value_counts().plot(kind='bar', color='#FD5C64', rot=0)
# plt.xlabel('totals_hits')
# sns.despine()
# plt.show()

# 4. totals_timeonsite: num
pd4 = pd.Series(ub_data_clean[:,4])
pd.Series.value_counts(pd4)



# 5. campaign_id: response
pd5 = pd.Series(ub_data_clean[:,5])
pd.Series.value_counts(pd5)
pd.Series(np.unique(pd5)).to_csv("unique_camp_id.csv")

# 6 clickinfo_slot: remove
pd6 = pd.Series(ub_data_clean[:,6])
pd.Series.value_counts(pd6)
pd6 = pd.get_dummies(pd6)

# 7 adnetworktype: remove
pd.Series.value_counts(pd.Series(ub_data_clean[:,7]))

# 8. source: all is "google", delete this var
pd.Series.value_counts(pd.Series(ub_data_clean[:,8]))

# 9 medium: all is "cpc", delete
pd.Series.value_counts(pd.Series(ub_data_clean[:,9]))

# 10 device_browser: 5
pd10 = pd.Series(ub_data_clean[:,10])
pd.Series.value_counts(pd10) # pick first leading 3
pd10 = pd.get_dummies(KeepFirstk_Rest2Other(pd10,3,"Other_device_browser"))


# 11. device_devicecategory: desktop, mobile, tablet
pd11 = pd.Series(ub_data_clean[:,11])
pd.Series.value_counts(pd11)
pd11 = pd.get_dummies(pd11)

# 12. device_operatingsystem: 4
pd12 = pd.Series(ub_data_clean[:,12])
pd.Series.value_counts(pd12)
pd12 = pd.get_dummies(KeepFirstk_Rest2Other(pd12,4,"Other_device_operatingsystem"))

# 13. device_javaenabled: all is False
pd.Series.value_counts(pd.Series(ub_data_clean[:,13]))

# 14. device_language: 1
pd14 = pd.Series(ub_data_clean[:,14])
pd.Series.value_counts(pd14)
pd14 = pd.get_dummies(KeepFirstk_Rest2Other(pd14,1,"Other_device_language"))

# 15 device_screencolors: 2
pd15 = pd.Series(ub_data_clean[:,15])
pd.Series.value_counts(pd15)
pd15 = pd.get_dummies(KeepFirstk_Rest2Other(pd15,2,"Other_device_screencolors"))


# 16 device_screenresolution: remove
pd.Series.value_counts(pd.Series(ub_data_clean[:,16]))

# 17: geo: set 30 dummy vars, the first 29 accounts 90%, the last is "Other"
pd17 = pd.Series(ub_data_clean[:,17])
pd.Series.value_counts(pd17)
print_cum_prop(pd.Series(ub_data_clean[:,17]))
pd17 = pd.get_dummies(KeepFirstk_Rest2Other(pd17,29,"Other_geo"))

# 18. hits_hitnumber: numeric
pd18 = pd.Series(ub_data_clean[:,18])
pd.Series.value_counts(pd18)


# 19. hits_hour: 3
pd19 = pd.Series(ub_data_clean[:,19])
pd.Series.value_counts(pd19)
for i in range(len(pd19)):
    if pd19[i] in range(0,8):
        pd19[i] = "hits_hour: 0-7"
    elif pd19[i] in range(8,16):
        pd19[i] = "hits_hour: 8-15"
    else:
        pd19[i] = "hits_hour: 16-23"
pd19 = pd.get_dummies(pd19)

# 20 hits_isinteraction: all is TRUE, remove it
pd.Series.value_counts(pd.Series(ub_data_clean[:,20]))

# 21 hits_minute: remove
pd.Series.value_counts(pd.Series(ub_data_clean[:,21]))

# 22 hits_time: numeric
pd22 = pd.Series(ub_data_clean[:,22])
pd.Series.value_counts(pd22)

# 23 hits_type:   PAGE : 131316 ,   EVENT : 4955
pd23 = pd.Series(ub_data_clean[:,23])
pd.Series.value_counts(pd23)
pd23 = pd.get_dummies(pd23)

# 24 hits_page_hostname: 3
pd24 = pd.Series(ub_data_clean[:,24])
pd.Series.value_counts(pd24)
# www.ticketmaster.com     87827
# m.ticketmaster.com       46006
# www1.ticketmaster.com     1614
# www.ticketmaster.ca        737
# m.ticketmaster.ca           87
pd24 = pd.get_dummies(KeepFirstk_Rest2Other(pd24,2,"Other_hits_page_hostname"))

# recall
# num: 0,2,3,4,18,22
# cat: 10,11,12,14,15,17,19,23,24
# resp: 5

final_pd = pd.concat([pd0,pd2,pd3,pd4,pd10,pd11,pd12,pd14,pd15,pd17,pd18,pd19,pd22,pd23,pd24],axis=1)

final_pd = final_pd.apply(ZeroOneScale,axis=1)

final_pd.to_csv("UserBehaviorX.csv")
pd5.index = date.index
fullvisitorid.index = date.index
visitid.index = date.index
customer_id.index = date.index

comp_id = pd.concat([pd5,date,fullvisitorid,visitid,customer_id],axis=1)
comp_id.to_csv("UserBehavior_comp_id_date.csv",header=False,index=False)

comp_id.shape
