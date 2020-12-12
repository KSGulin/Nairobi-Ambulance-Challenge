import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

def Cross_Validation_Single(model, X, Y, K = 5):
#	Performs a single iteration of K-Fold cross validation with a single model

	skf = KFold(n_splits=K, shuffle = True)

	performance = 0
	for train_idx, val_idx in skf.split(X):
		model.Train(X[train_idx], Y[train_idx])
		performance += model.Eval(X[val_idx], Y[val_idx])

	performance = performance/K

	return performance

def Cross_Validation_Tune(modelclass, X, Y, params, K = 5, upsample = 0):
	performance = np.zeros(len(params))
	models = []

	for i,comb in enumerate(params):
		model = modelclass(comb, upsample=upsample)
		models.append(model)
		performance[i] = Cross_Validation_Single(model, X, Y, K)

	best_m = np.argmax(performance)

	return models[best_m], performance[best_m]

def Param_Permutations(params):
	return list(itertools.product(*params))

def Cross_Validate_Multiple(modelclass, X, Y, params, K= 5, upsample = 0):
	models = []
	performances = []
	for i in range(len(Y)):
		model, performance = Cross_Validation_Tune(modelclass, X[i], Y[i], params, K = K, upsample = upsample)
		models.append(model)
		performances.append(performance)
		marginal_performance = np.sum(model.model.predict(X[i]) == Y[i])/len(Y[i]) - (1 - np.sum(Y[i])/len(Y[i]))
		relative_improvement = marginal_performance*len(Y[i])/(np.sum(Y[i]))
		print("Model trained outperforms guessing no crashes on full data set by " + str(marginal_performance))
		print('This correspond to ' + str(relative_improvement*100) + '% of possible improvement over baseline')
		skf = KFold(n_splits=5, shuffle = True)

		relative_improvement_val = 0
		marginal_performance_val = 0
		for n in range(5):
			for train_idx, val_idx in skf.split(X[i]):
				model.Train(X[i][train_idx], Y[i][train_idx])
				marginal_performance = (np.sum(model.model.predict(X[i][val_idx]) == Y[i][val_idx])/len(Y[i][val_idx]) - (1 - np.sum(Y[i][val_idx])/len(Y[i][val_idx])))/(np.sum(Y[i][val_idx]))
				marginal_performance_val += marginal_performance
				relative_improvement_val += marginal_performance*len(val_idx)/(np.sum(Y[i][val_idx]))

		print("Model trained outperforms guessing no crashes on validation data by " + str(marginal_performance_val/25))
		print(' This correspond to ' + str(relative_improvement_val*100/25) + '% of possible improvement over baseline')

	return models, performances

def Cross_Validate_Regression(modelclass, X, Y, params, K= 5, upsample = 0):

	models = []
	performances = []
	for i in range(len(Y)):
		model, performance = Cross_Validation_Tune(modelclass, X[i], Y[i], params, K = K, upsample = upsample)
		models.append(model)
		performances.append(performance)
		skf = KFold(n_splits=5, shuffle = True)

		MSE = 0
		baseline_MSE = 0
		for n in range(5):
			for train_idx, val_idx in skf.split(X[i]):
				model.Train(X[i][train_idx], Y[i][train_idx])
				MSE += np.mean(np.square(model.model.predict(X[i][val_idx]) - Y[i][val_idx]))
				baseline_MSE += np.mean(np.square(Y[i][val_idx]-np.mean(Y[i][val_idx])))

		print('MSE for the best model is ' + str(MSE/25))
		print('This outperforms the baseline model by ' + str((baseline_MSE - MSE)/25))

	return models, performances

def Model_Multiple_CV(model, X, Y):
	skf = KFold(n_splits=5, shuffle = True)
	relative_improvement_val = 0
	marginal_performance_val = 0
	for n in range(5):
		for train_idx, val_idx in skf.split(X):
			model.Train(X[train_idx], Y[train_idx])
			marginal_performance = (np.sum(model.model.predict(X[val_idx]) == Y[val_idx])/len(Y[val_idx]) - (1 - np.sum(Y[val_idx])/len(Y[val_idx])))/(np.sum(Y[val_idx]))
			marginal_performance_val += marginal_performance
			relative_improvement_val += marginal_performance*len(val_idx)/(np.sum(Y[val_idx]))

def Estimate_Performance_Reg(model, X, Y):
	skf = KFold(n_splits=5, shuffle = True)
	MSE = []
	for n in range(5):
		for train_idx, val_idx in skf.split(X):
			model.Train(X[train_idx], Y[train_idx])
			MSE.append(np.mean(np.square(model.model.predict(X[val_idx]) - Y[val_idx])))

	return np.asarray(MSE)
