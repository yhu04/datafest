import csv
import numpy as np  
import pandas as pd 

#load user data
path_user = "UserBehavior_comp_id_date.csv"
user_data = np.genfromtxt(path_user,delimiter=",",dtype=object)
#campaignID,time,userID

#load ad data
path_ad = "c.csv"
ad_dict = {}
with open(path_ad) as ad_file:
	reader = csv.reader(ad_file)
	for row in reader: 
		new_time = int(row[2].replace('-',''))
		if row[1] not in ad_dict:
			print row[1]
			ad_dict[row[1]] = [(new_time,row[3])]
		else:
			temp = ad_dict[row[1]]
			temp.append((new_time,row[3]))
			ad_dict[row[1]] = temp 

new_data = []
index = [] 

for i in range(144671):
	try:
		campaign_id = user_data[i,0]
		time = float(user_data[i,1])
		if campaign_id in ad_dict:
			min_diff = float("inf") 
			right_label = float("inf") 

			for item in ad_dict[campaign_id]:
				if item[0]<time and time-item[0]<min_diff:
					min_diff = item[0]-time
					right_label = item[1]
			temp2 = list(user_data[i,:])
			temp2.append(right_label)
			new_data.append(temp2)
			index.append(i)
	except:
		print ""

label_matrix = {}
for i in range(len(new_data)):
	try:
		if new_data[i][2] in label_matrix:
			label = int(new_data[i][5])
			if label_matrix[new_data[i][2]][label]!=1:
				label_matrix[new_data[i][2]][label]=1
		else:
			label = int(new_data[i][5])
			label_matrix[new_data[i][2]]=[0,0,0,0,0,0]
			label_matrix[new_data[i][2]][label]=1
	except:
		print ""

#label_matrix.values()
print new_data[1]
user_label = pd.DataFrame(label_matrix.values())
user_label.to_csv("user_label.csv", sep=',')
index_user = pd.DataFrame(list(label_matrix.keys()))
index_user.to_csv("user_id_order.csv",sep=",")
#np.savetxt("user_label.csv", user_label, delimiter=",")
#np.savetxt("user_id_order",np.array(label_matrix.keys(),dtype=object),delimiter=",")