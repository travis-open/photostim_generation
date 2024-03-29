import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import ndimage
import random
import datetime

##functions to build stimsets and optimize order to avoid nearby locations
##2-opt algorithm from https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy



def draw_rect(n_cols, n_rows, i_col, i_row, img_w=1024, img_h=1024, plot = False):
	'''
	Draws a single rectangle at column and row indexes.
	Intended to be used in camera coordinates, then translated to DMD.
	'''
    ##margins correspond to region of camera image covered by DMD. Only relevant for 1024x1024.
    ##maybe require 1024x1024 image until better solution determined
	left_margin = 148 
	right_margin =268 
	upper_margin =168 
	lower_margin = 336
	sub_w = img_w - left_margin - right_margin
	sub_h = img_h - upper_margin - lower_margin
	rect_width = math.floor(sub_w/n_cols) - 1
	rect_height = math.floor(sub_h/n_rows) - 1
	x0 = left_margin + i_col*rect_width
	y0 = upper_margin + i_row*rect_height
	stim = cv2.rectangle(
	img = np.zeros((img_w, img_h), dtype = np.uint8),
	rec = (x0, y0, rect_width, rect_height),
	color = 255, 
	thickness = -1)
	if plot:
		plt.imshow(stim)
	return stim

def draw_all_grid_rects(n_cols, n_rows, img_w=1024, img_h=1024):
	'''
	Creates 3d numpy array of all possible grid locations
	'''
	n_stims = n_cols * n_rows
	grid_stims = np.zeros((img_w, img_h, n_stims), dtype=np.uint8)
	grid_i = 0
	for ic in np.arange(n_cols):
		for ir in np.arange(n_rows):
			stim = draw_rect(n_cols, n_rows, ic, ir)
			grid_stims[:,:,grid_i] = stim
			grid_i += 1
	return grid_stims


def image_sequence_com(image_sequence):
	'''
	Return a list of center of masses [(x,y) coordinates] for each image in sequence
	'''
	n_images = image_sequence.shape[2]
	com_list = []
	for i in range(n_images):
		image = image_sequence[:,:,i]
		com = ndimage.center_of_mass(image)
		com_list.append(com)
	return com_list

def distance_between_coms(i, j, coms):
	'''
	compute distance between two elements of center of mass list
	'''

	(ix, iy) = coms[i]
	(jx, jy) = coms[j]
	xdist = jx - ix
	ydist = jy - iy
	distance = (xdist**2 + ydist**2)**0.5
	return distance

def distances_all_steps(sequence_order, coms):
	'''
	return np array of distances of all steps
	'''

	n_stims = len(sequence_order)
	distances = np.zeros(n_stims - 1)
	for step in np.arange(0, (n_stims-1)):
		i, j = sequence_order[step], sequence_order[step+1]
		d = distance_between_coms(i, j, coms)
		distances[step] = d
	return distances

def distances_n_steps(sequence_order, coms, n_steps=2):


    n_stims = len(sequence_order)
    distances = np.zeros(n_stims - n_steps)
    for step in np.arange(0, (n_stims-n_steps)):
        i, j = sequence_order[step], sequence_order[step + n_steps]
        d = distance_between_coms(i, j, coms)
        distances[step] = d
    return distances

##https://stackoverflow.com/questions/25585401/travelling-salesman-in-scipy
##suggested solution to traveling salesman problem, aiming to minimize route distance
def two_opt(cities,improvement_threshold): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
	route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
	improvement_factor = 1 # Initialize the improvement factor.
	best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
	while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
		distance_to_beat = best_distance # Record the distance at the beginning of the loop.
		for swap_first in range(1,len(route)-2): # From each city except the first and last,
			for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
				new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
				new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
				if new_distance < best_distance: # If the path distance is an improvement,
					route = new_route # make this the accepted best route
					best_distance = new_distance # and update the distance corresponding to this route.
		improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
	return route # When the route is no longer improving substantially, stop searching and return the route.

def path_distance(r, c):
	return np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])


def two_opt_swap(r, i, k):
	return np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

## modified 2-opt to find longest route to visit all stim locations
def two_opt_long(coms,improvement_threshold, r_seed = 42): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
	route = np.arange(coms.shape[0]) # Make an array of row numbers corresponding to coms.
	random.seed(r_seed)
	random.shuffle(route) ##starting with 0...n seemed to result in quandrant-ish patterns
	improvement_factor = 1 # Initialize the improvement factor.
	best_distance = path_distance(route,coms) # Calculate the distance of the initial path.
	while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
		distance_to_beat = best_distance # Record the distance at the beginning of the loop.
		for swap_first in range(1,len(route)-2): # From each city except the first and last,
			for swap_last in range(swap_first+1,len(route)): # to each of the coms following,
				new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these coms
				new_distance = path_distance(new_route,coms) # and check the total distance with this modification.
				min_dist = np.min(distances_all_steps(new_route, coms))
				if new_distance > best_distance: # If the path distance is an improvement,
					route = new_route # make this the accepted best route
					best_distance = new_distance # and update the distance corresponding to this route.
		improvement_factor = best_distance/distance_to_beat -1 # Calculate how much the route has improved.
	return route # When the route is no longer improving substantially, stop searching and return the route.

def two_opt_one_two_step(coms,improvement_threshold, r_seed = 42): # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
	route = np.arange(coms.shape[0]) # Make an array of row numbers corresponding to coms.
	random.seed(r_seed)
	random.shuffle(route) ##starting with 0...n seemed to result in quandrant-ish patterns
	improvement_factor = 1 # Initialize the improvement factor.
	best_distance = np.min(distances_all_steps(route, coms)) + np.min(distances_n_steps(route, coms, n_steps=2)) # consider minimum 1-step and 2-step sizes
	while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
		distance_to_beat = best_distance # Record the distance at the beginning of the loop.
		for swap_first in range(1,len(route)-2): # From each city except the first and last,
			for swap_last in range(swap_first+1,len(route)): # to each of the coms following,
				new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these coms
				new_distance = np.min(distances_all_steps(new_route, coms)) + np.min(distances_n_steps(new_route, coms, n_steps=2)) # and check the total distance with this modification.
				if new_distance > best_distance: # If the path distance is an improvement,
					route = new_route # make this the accepted best route
					best_distance = new_distance # and update the distance corresponding to this route.
		improvement_factor = best_distance/distance_to_beat -1 # Calculate how much the route has improved.
	return route # When the route is no longer improving substantially, stop searching and return the route.

def polygon(sides, radius=1, rotation=0, translation=None):
	one_segment = math.pi * 2 / sides
	points = [
		(math.sin(one_segment * i + rotation) * radius,
		math.cos(one_segment * i + rotation) * radius)
		for i in range(sides)]

	if translation:
		points = [[sum(pair) for pair in zip(point, translation)]
		for point in points]
	return points

def draw_hex(n_cols, n_rows, i_col, i_row, img_w = 1024, img_h = 1024, plot = False):
	left_margin = 148 
	right_margin =268 
	upper_margin =168 
	lower_margin = 336
	sub_w = img_w - left_margin - right_margin
	sub_h = img_h - upper_margin - lower_margin
	poly_r = math.floor(sub_w/n_cols)/2
	one_segment = math.pi * 2 / 6
	y_shift = (math.cos(one_segment*1)*poly_r)+poly_r+1 ##+1 to err on side of avoid overlap
	x_shift = (math.sin(one_segment*1)*poly_r)*2+1
	x0 = left_margin + i_col*x_shift + x_shift
	y0 = upper_margin + i_row*y_shift +y_shift
	if i_row%2==0:
		x0+=x_shift/2
	points = polygon(6, radius = poly_r, translation = [x0, y0])
	points = np.array(points).astype(np.int32).reshape((-1, 1, 2))
	stim = cv2.fillPoly(img = np.zeros((img_w, img_h), 
		dtype = np.uint8),
		pts = [points],
		color = 255)
	if plot:
		plt.imshow(stim)
	return stim

def draw_all_grid_hex(n_cols, n_rows, img_w=1024, img_h=1024):
	n_stims = n_cols * n_rows
	grid_stims = np.zeros((img_w, img_h, n_stims), dtype = np.uint8)
	grid_i = 0
	for ic in np.arange(n_cols):
		for ir in np.arange(n_rows):
			stim = draw_hex(n_cols, n_rows, ic, ir)
			grid_stims[:, :, grid_i] = stim
			grid_i += 1
	return grid_stims

class StimSequenceSet():
    def __init__(self, image_sequence, name, sequence_dict):
        self.image_sequence = image_sequence
        self.name = name
        self.n_patterns = image_sequence.shape[2]
        self.sequence_dict = sequence_dict
        self.time_created = datetime.datetime.now()
        self.image_shape = image_sequence.shape[:-1]

    def get_single_image(self, i):
        return self.image_sequence[:,:,i]
    
    def get_ordered_seq(self, order):
        new_stimset = np.zeros(self.image_shape+(len(order),),
        	dtype=np.uint8)
        for i in range(len(order)):
            stim_index = order[i]
            stim_pattern = self.get_single_image(stim_index)
            new_stimset[:,:,i] = stim_pattern
        return new_stimset
        
    def get_ordered_seq_by_name(self, order_name):
        order = self.get_order_by_name(order_name)
        new_stimset = self.get_ordered_seq(order)
        return new_stimset

    def get_default_seq(self):
    	new_stimset = self.get_ordered_seq_by_name('default')
    	return new_stimset

    def get_order_by_name(self, order_name):
    	assert order_name in self.sequence_dict, f"The order {order_name} is unknown for this image sequence."
    	order = self.sequence_dict[order_name]
    	return order

    def get_summed_image(self, image_list):
    	summed_image = np.zeros(self.image_shape, dtype = np.uint8)
    	for i in image_list:
    		summed_image += self.image_sequence[:, :, i]
    	return summed_image