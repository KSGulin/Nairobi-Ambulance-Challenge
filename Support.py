import numpy as np
import pandas as pd


##### Data Read & Write ##################################

def Load_Crash_Data():
#	Read Crash data from .csv file and convert to numpy array

	df = pd.read_csv('Data/Train.csv', parse_dates=['datetime'])
	array = np.delete(df.to_numpy(),0, 1)
	return array


def Print_Sub_File(locations):
#	Prints a competition submission file given a 2D matrix of ambulance locations
#
#	Inputs:
#	locations - 2D array of abulance locations
#		Assumptions:
#		Currently assumed that ambulance locations are fixed
	
	ss = pd.read_csv('Data/SampleSubmission.csv', parse_dates=['date'])
	ss.head()
	merged = list(zip(locations[:,0],locations[:,1]))
	locations_flat = [c for couple in merged for c in couple ]
	d = pd.DataFrame([])
	d['date'] = ss['date'] 
	for i in range(len(locations_flat)):
		d[ss.keys()[i+1]] = locations_flat[i]
	d.head()
	d.to_csv('Data/submission.csv')


##### Other Functions ######################################

def Competition_Metric(X, Y, time_dep):
# 	Carries out calculation of performance metric used in the competition, 
# 	which is simply the sum of 2D Euclidean distances from each crash (in Y) to the nearest ambulance (in X)
# 
#	Inputs:
#	X - list of 2D numpy arrays of locations of ambulances (in lat, lon degrees) with associated time stamps 
#		Assumption: 
#		If time_dep == 0, then X is a list with only one 2D aray
#		The numpy array has entries of type [Timestamp, double, double]
#		Timestamp marks the beginning of the time period (e.g. 0:00 - 3:00 will be denoted by 0:00)
#
#	Y - 2D numpy array of locations of crashes (in lat, lon degrees) with associated time stamps
#		Assumptions: 
#		Formatted as numpy array with entries of type [Timestamp, double, double]
#
#	time_dep - Flag for whether the input X data is specified for every time step or is assumed to be constant
#
	
	total_dist = 0

	if time_dep:
		for y in Y:
			for x in X:
				if y[0] > x[0][0]:
					min_dist = 100
					for i in range(6):
						t_dist = np.linalg.norm(x - y[1:])
						if t_dist < min_dist:
							min_dist = t_dist
					total_dist += min_dist

	else:
		X = X[0]
		for y in Y:
			for x in X:
				min_dist = 100
				t_dist = np.linalg.norm(x - y[1:])
				if t_dist < min_dist:
					min_dist = t_dist

			total_dist += min_dist


	return total_dist
				