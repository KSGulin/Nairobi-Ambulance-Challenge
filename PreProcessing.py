import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from sklearn.utils import resample
from scipy import stats


class cell:
	def __init__(self, left_lat, bottom_lon, d):
		self.left_lat = left_lat
		self.right_lat = left_lat + d
		self.bottom_lon = bottom_lon
		self.top_lon = bottom_lon + d
		self.width = d
		self.crashes = []

	def addcrash(self, x):
		if self.isin(x[1:]):
			self.crashes.append(x)
			return 1
		else:
			return 0

	def isin(self, x):
		if x[1] > self.left_lat and x[1] < self.right_lat:
			if x[0] > self.bottom_lon and x[0] < self.top_lon:
				return 1
		return 0
	

def Merge_Roads_Crashes(segments_merged, crash_locations, crashes, t = .0045):
	min_distances = np.zeros(crash_locations.shape[0])
	min_segments = np.zeros(crash_locations.shape[0])
	min_dist = 100
	for i, loc in enumerate(crash_locations):
		min_dist = 100
		min_seg = -1
		for j in range(segments_merged.shape[0]):
			t_dist = segments_merged['geometry'][j].distance(Point(loc))
			if t_dist < min_dist:
				min_dist = t_dist
				min_seg = j
		min_distances[i] = min_dist
		min_segments[i] = min_seg
		
	segments_merged.insert(3, 'crashes', 0)
	segments_merged.insert(4, 'crash_coords', 0)
	segments_merged.insert(5, 'crash_times', 0)
	segments_merged['crashes'] = np.empty((len(segments_merged), 0)).tolist()
	segments_merged['crash_coords'] = np.empty((len(segments_merged), 0)).tolist()
	segments_merged['crash_times'] = np.empty((len(segments_merged), 0)).tolist()
	
	inds_unmerged=[]

	for i, s in enumerate(min_segments):
		if min_distances[i] < t:
			id = segments_merged['segment_id'][s]
			inds = np.where([segments_merged['segment_id'] == id])[1]
			for ind in inds:
				segments_merged.iloc[ind, 3].append(1/len(inds))
				segments_merged.iloc[ind, 4].append(crash_locations[i])
				segments_merged.iloc[ind, 5].append(crashes[i][0])
		else:
			inds_unmerged.append(i) 
			
	return segments_merged, inds_unmerged
	
def Create_Paired_Series(segments_merged, weather):
	predictive_data = np.zeros([len(weather)*len(segments_merged), 233])
	crash_labels = np.zeros([len(segments_merged), len(weather)])
	for i, rrow in segments_merged.iterrows():
		set = np.zeros([len(weather), 233])
		for j, wrow in weather.iterrows():
			set[j] = np.concatenate((wrow[1:].to_numpy(), rrow[6:].to_numpy()), axis = 0)
			for k, crash in enumerate(rrow[3]):
				if rrow[5][k].date() == wrow[0].date():
					crash_labels[i, j] += rrow[3][k]
		predictive_data[i*len(weather):(i+1)*len(weather)] = set
		
	crash_labels = crash_labels.flatten()
	return predictive_data, crash_labels


def Create_Paired_Series2(segments_merged):
	predictive_data = np.zeros([8*len(segments_merged), 228])
	crash_labels = np.zeros([len(segments_merged), 8])
	for i, rrow in segments_merged.iterrows():
		for h in range(8):
			predictive_data[i*8 + h] = np.concatenate(([h],  rrow[6:].to_numpy()), axis = 0)
		for k, crash in enumerate(rrow[5]):
			h_ind = int(np.floor(crash.hour/3))
			crash_labels[i,h_ind] += rrow[3][k]
		
	crash_labels = crash_labels.flatten()
	return predictive_data, crash_labels
		
def Create_Grid(crashes, d):
	lonmax = np.max(crashes[:,1]) + .05
	lonmin = np.min(crashes[:,1]) - .05
	latmax = np.max(crashes[:,2]) + .05
	latmin = np.min(crashes[:,2]) - .05
	nlon = np.floor((lonmax  - lonmin)/d) + 1
	nlat = np.floor((latmax - latmin)/d) + 1

	grid = []
	for i in range(int(nlat)):
		grid.append([])
		for j in range(int(nlon)):
			grid[i].append(cell(latmin+(d*i), lonmin+(d*j), d))

	for crash in crashes:
		j = int(np.floor((crash[1] - lonmin)/d))
		i = int(np.floor((crash[2] - latmin)/d))
		ret = grid[i][j].addcrash(crash)
		if (ret == 0):
			print('crash outside cell range')

	return grid

def Create_Unpaired_Series(grid, weather):
	predictive_data = np.zeros([len(grid), 8*len(weather), 7])
	crash_labels = np.zeros([len(grid), 8*len(weather)])
	ind =[]
	active_cells = []
	for i, cell in enumerate(grid):
		if (cell.crashes):
			active_cells.append(i)
			for j, wrow in weather.iterrows():
				for h in range(8):
					predictive_data[i,j*8 +h] = np.concatenate(([h],  wrow[1:].to_numpy()), axis = 0)
				for k, crash in enumerate(cell.crashes):
					if crash[0].date() == wrow[0].date():
						h_ind = int(np.floor(crash[0].hour/3))
						crash_labels[i, j*8 +h_ind] += 1
		else:
			ind.append(i)

	predictive_data = np.delete(predictive_data, ind, 0)
	#crash_labels = crash_labels.flatten()
	crash_labels = np.delete(crash_labels, ind, 0)
	return predictive_data, crash_labels, active_cells

def Upsample(X, Y, r = 1):
	ind0 = np.argwhere(Y == 0)
	ind1 = np.argwhere(Y == 1)
	if len(ind1) > 0:
		X1_resampled = resample(X[ind1], replace=True, n_samples=int(r*len(ind0)))
		X = np.concatenate((X[ind0], X1_resampled)).squeeze()
		Y = np.concatenate((Y[ind0], np.ones((int(r*(len(ind0))), 1)))).squeeze()
	return X, Y

#def Replace_Nan_Negative(X):

def Replace_Nan_Mode(X):
	replaced_nans = {}
	for i in range(X.shape[1]):
		mode = stats.mode(X[:,i])
		nans = np.isnan(X[:,i])
		if np.any(nans):
			X[nans,i] = mode[0]
			replaced_nans[i] = nans.astype(int)
	
	return X, replaced_nans
	