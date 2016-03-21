import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics 
from sklearn.decomposition import PCA
import seaborn as sns 

def load_data(path_ad, path_id):
	ad_words = np.genfromtxt(path_ad,delimiter=",",dtype=object)
	campaign_ID = np.genfromtxt(path_id,delimiter=",",dtype=object)
	return ad_words, campaign_ID

def cluster(ad_words,k_min,k_max):
	# determin k range
	k_range = range(k_min,k_max)
	# fit the kmeans model for each n_clusters = k
	max_score = 0 
	max_k = k_range[0]
	for k in k_range:
		k_means_model = KMeans(n_clusters=k)
		k_means_model.fit(ad_words)
		sample_size = int(len(ad_words)*0.01)
		print "Finish fit"
		k_metric = metrics.silhouette_score(ad_words,k_means_model.labels_,metric='euclidean',sample_size=sample_size)
		print k_metric,k
		if k_metric > max_score:
			max_score = k_metric
			max_k = k 
	return max_score,max_k

def fit_model(ad_words, max_k):
	best_model = KMeans(n_clusters=max_k)
	best_model.fit(ad_words)
	campaign_label = best_model.labels_
	return campaign_label


def plot_result(ad_words,max_k):
	ad_words_reduced = PCA(n_components=2).fit_transform(ad_words)
	best_model = KMeans(n_clusters=max_k)
	best_model.fit(ad_words_reduced)

	#plot decision boundary
	h = .02 
	x_min, x_max = ad_words_reduced[:, 0].min() - 1, ad_words_reduced[:, 0].max() + 1
	y_min, y_max = ad_words_reduced[:, 1].min() - 1, ad_words_reduced[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.PuBu,
           aspect='auto', origin='lower')

	#plt.contourf(xx, yy, Z, cmap=plt.cm.PuBu)
	#plt.axis('off')
	#plt.plot(ad_words_reduced[:, 0], ad_words_reduced[:, 1],'k.')
	print "Finish plot data"

	plt.title('Advertisement Segmentation - Decision Boundary')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.xlabel('component1')
	plt.ylabel('component2')
	plt.savefig("Decision Boundary-PCA.jpg")

def scatter_cluster(ad_words,max_k):
	ad_words_reduced = PCA(n_components=2).fit_transform(ad_words)
	best_model = KMeans(n_clusters=max_k)
	best_model.fit(ad_words_reduced)	

	plt.figure(1)
	plt.clf()

	plt.plot(ad_words_reduced[:, 0], ad_words_reduced[:, 1],'k.', markersize=1)

	plt.title('Advertisement Segmentation')
	plt.xlim(ad_words_reduced[:,0].min(), ad_words_reduced[:,0].max())
	plt.ylim(ad_words_reduced[:,1].min(), ad_words_reduced[:,1].max())
	plt.xlabel('component1')
	plt.ylabel('component2')
	plt.savefig("cluster_scatter.jpg")

def main():
	# load data 
	path_ad = "out.csv"
	path_id = "camp_id.csv"
	ad_words,campaign_ID = load_data(path_ad,path_id)
	print "load data"

	# cluster parameter 
	#k_min=2
	#k_max=11
	#max_score, max_k = cluster(ad_words,k_min,k_max)
	#print max_score, max_k

	# fit model 
	max_k = 6

	#visualization 
	plot_result(ad_words,max_k)

	#campaign_label = fit_model(ad_words,max_k)
	#print campaign_label.shape
	#print campaign_ID.shape
	#lable_data = np.column_stack((campaign_label,campaign_ID[:,1]))
	#lable_data=lable_data[~np.isnan(lable_data).any(axis=1)]
	#np.savetxt("campaign_label.csv", lable_data, delimiter=",",fmt="%s")

if __name__ == '__main__':
	main()


