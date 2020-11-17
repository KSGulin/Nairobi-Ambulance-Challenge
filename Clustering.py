from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import Support as Sup

class Kmeans:
#	Simple K-Means approach for ambulance placement
#	The flag "plot" is used to turn on/off plotting

	def __init__(self, plot = 0):
		self.Eval_Fun = Sup.Competition_Metric
		self.centers = []
		self.plot = plot
	
	def Train(self, X, Y):
		kmeans = KMeans(n_clusters = 6, init = 'k-means++', n_init = 5, algorithm = 'full').fit(X)
		self.centers =  kmeans.cluster_centers_

		if self.plot:
			c_pred = kmeans.predict(X)
			plt.scatter(X[:, 0], X[:, 1], c=c_pred)
			plt.scatter(self.centers[:,0], self.centers[:,1], marker = 'x' ,c= 'k')
			plt.show()

	def Eval(self, X, Y):
		return self.Eval_Fun(self.centers, Y, 0)
	

	

