import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

def Merge_Roads_Crashes(segments_merged, crash_locations, t = .0045):
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
	segments_merged.insert(4, 'crash_coords', [])
	
	for i, s in enumerate(min_segments):
		if min_distances[i] < t:
			id = segments_merged['segment_id'][s]
			inds = np.where([segments_merged['segment_id'] == id])[1]
			segments_merged.iloc[inds, 3] += 1/len(inds)
			segments_merged.iloc[inds, 4].append(crash_locations[i])
			
	return segments_merged