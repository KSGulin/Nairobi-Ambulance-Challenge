import pandas as pd
from sklearn import feature_selection

def Nan_Threshold(X, t):
	if X.isnull().sum()/len(X) > t:
		return 1
	else:
		return 0
		
def Variance_Threshold(X, t):
	sel = feature_selection.VarianceThreshold(t)
	return = sel.fit_transform(X)
		
def Top_Variance_Select(X, n):
	sel = feature_selection.VarianceThreshold()
	variances = sel.fit(X).variances_
	inds = np.argpartition(variances, -n)[-n:]
	return X[:,inds]