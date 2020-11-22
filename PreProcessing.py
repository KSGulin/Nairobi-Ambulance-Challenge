import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

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
	
	
	for i, s in enumerate(min_segments):
		if min_distances[i] < t:
			id = segments_merged['segment_id'][s]
			inds = np.where([segments_merged['segment_id'] == id])[1]
			for ind in inds:
				segments_merged.iloc[ind, 3].append(1/len(inds))
				segments_merged.iloc[ind, 4].append(crash_locations[i])
				segments_merged.iloc[ind, 5].append(crashes[i][0])
			
	return segments_merged
	
def Create_Time_Series_Data(segments_merged, weather):
	predictive_data = set = np.zeros([len(weather)*len(segments_merged), 233])
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
		