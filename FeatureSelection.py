import pandas as pd
import numpy as np
from sklearn import feature_selection

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