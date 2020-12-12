import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.metrics import confusion_matrix
import scipy.stats as ss

def Nan_Threshold(X, t):
	n = X.shape[0]
	X_out = X
	for i in range(X.shape[1]):
		if np.sum(np.isnan(X[:,i]))/n > t:
			X_out = np.delete(X_out, i, 1)
	return X_out
		
def Variance_Threshold(X, t):
	sel = feature_selection.VarianceThreshold(t)
	return sel.fit_transform(X)
		
def Top_Variance_Select(X, n):
	sel = feature_selection.VarianceThreshold()
	variances = sel.fit(X).variances_
	inds = np.argpartition(variances, -n)[-n:]
	return X[:,inds]

#### Note most of this function was pulled from online (Stackoverflow)
def Calculate_Cramers_Corrected(x, y):
	result=-1
	if len(np.unique(x))==1 :
		print("First variable is constant")
	elif len(np.unique(y))==1:
		print("Second variable is constant")
	else:   
		conf_matrix=pd.crosstab(x, y)

		if conf_matrix.shape[0]==2:
			correct=False
		else:
			correct=True

		chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]

		n = sum(conf_matrix.sum())
		phi2 = chi2/n
		r,k = conf_matrix.shape
		phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))	
		rcorr = r - ((r-1)**2)/(n-1)
		kcorr = k - ((k-1)**2)/(n-1)
		result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
	return round(result,6)