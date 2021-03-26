import numpy as np

boolvec_dim = 5
rep_bools_len = boolvec_dim + 1 # Set this to how many rep boolean vectors to use in the train set.

# Returns a weighted sum of value times one-indexed index.
def get_rotation_amount(bool_vec):
	rot_amt = sum([i * val for i, val in enumerate(bool_vec, start=1)])
	return rot_amt

# Given classes and rotation amount, rotate class under mod num_classes.
def rotate_class(classes, rot_amts, num_classes):
	assert max(classes) < num_classes and min(classes) >= 0, "Classes are out of range."
	assert len(rot_amts) == len(classes), "Classes and rot_amts need to be same length."

	return (classes + rot_amts) % num_classes

# Given classes and multiplication amount, multiply each class by that amount under mod num_classes.
def mod_mult(classes, mult_amts, num_classes):
	assert max(classes) < num_classes and min(classes) >= 0, "Classes are out of range."
	assert len(mult_amts) == len(classes), "Classes and mult_amts need to be same length."

	return (classes * mult_amts) % num_classes

# Return an array of random boolean vectors, where each row is a boolean vector.
def get_rand_bool_vecs(arr_size, boolvec_dim):
	return np.random.randint(2, size=(arr_size, boolvec_dim))

# Returns D + 1 boolean vectors, where each vector is 1 flip away from at least one other vector.
# This provides enough boolean vectors to understand what each bit is doing for the data.
def get_representative_bools(boolvec_dim):
	bools = [tuple(np.random.randint(2, size=boolvec_dim).tolist())]
	for i in range(boolvec_dim):
		rand_bool = list(bools[np.random.randint(len(bools))]).copy()
		rand_bool[i] = 1 - rand_bool[i]
		bools.append(tuple(rand_bool))

	if len(bools) < rep_bools_len:
		all_bools = get_all_bitstrings(boolvec_dim)
		curr_bools = set(bools)
		for poss_bool in all_bools:
			if len(bools) == rep_bools_len: break
			
			if poss_bool not in curr_bools:
				curr_bools.add(poss_bool)
				bools.append(poss_bool)

	assert len(np.unique(bools, axis=0)) == rep_bools_len, "Not enough rep bools."
	return bools

# Returns row intersection between arr1 and arr2
def intersect2D(arr1, arr2):
	return np.array([x for x in set(tuple(x) for x in arr1) & set(tuple(x) for x in arr2)])

# Returns arr_size array of representative boolean vectors.
def get_rep_bool_vecs(arr_size, boolvec_dim, rep_bools):
	assert len(rep_bools) <= arr_size, "Desired array size is less than the number of rep_bools."
	assert rep_bools_len == len(rep_bools), "Mismatch between boolvec_dim and dim of rep_bools."

	# This tries to fit as many multiples of rep_bools into arr_size.
	bool_vecs = np.tile(rep_bools, ((int)(np.floor(arr_size/(rep_bools_len))), 1))

	# Then for the remaining vectors left, randomly select from rep_bools to fill up the array.
	if arr_size > len(bool_vecs):
		rep_bools = np.array(rep_bools)
		np.random.shuffle(rep_bools)
		rand_rep_bools = rep_bools[:arr_size - len(bool_vecs)].reshape(-1, boolvec_dim)
		bool_vecs = np.concatenate((bool_vecs, rand_rep_bools), axis=0)

	np.random.shuffle(bool_vecs)
	assert len(intersect2D(rep_bools, np.unique(bool_vecs, axis=0))) == len(rep_bools), \
					"Not all the rep_bools were used."

	return bool_vecs

# Returns the hamming distance between two vectors.
# This quantifies the number of flips to get from one to the other, and is symmetric.
def hamming_distance(vec1, vec2):
	return np.count_nonzero(np.array(vec1) != np.array(vec2))

# Generates all bitstring vectors, with length boolvec_dim.
def get_all_bitstrings(boolvec_dim):
	n = boolvec_dim
	bitstrings = []
	for i in range(2 ** n, 2 ** (n + 1)):
		bitmask = bin(i)[3:]
		vec = tuple([int(c) for c in list(bitmask)])
		bitstrings.append(vec)

	assert len(np.unique(bitstrings, axis=0)) == np.power(2, boolvec_dim), \
					"Not enough bitstrings."
	return bitstrings

# Grabs all the bitstrings that are "dist" away from closest vec in rep_bools.
# exlude_train_bool flag will signal if we want to keep any of the vecs
# that were in our training data or not.
def get_neighbor_bools(rep_bools, boolvec_dim, dist, exclude_train_bools=True):
	assert rep_bools_len == len(rep_bools), "Mismatch between boolvec_dim and dim of rep_bools."
	all_bools = get_all_bitstrings(boolvec_dim)
	neighbors = []
	if exclude_train_bools:
		all_bools = [list(x) for x in list(set(all_bools) - set(rep_bools))]

	for poss_vec in all_bools:
		poss_dist = min([hamming_distance(poss_vec, x) for x in rep_bools])
		if poss_dist == dist:
			neighbors.append(poss_vec)

	return neighbors

# Returns arr_size array of all boolean vectors, dist away from train_bools
def get_dist_bool_vecs(arr_size, boolvec_dim, rep_bools, dist, exclude_train_bools=True):
	assert rep_bools_len == len(rep_bools), "Mismatch between boolvec_dim and dim of rep_bools."

	neighbor_bools = get_neighbor_bools(rep_bools, boolvec_dim, dist, exclude_train_bools)
	assert len(neighbor_bools) != 0, "No neighboring boolean vectors!"
	assert len(neighbor_bools) <= arr_size, "Desired array size is less than number of neighbor vectors."

	# This tries to fit as many multiples of neighbor_bools into arr_size.
	bool_vecs = np.tile(neighbor_bools, ((int)(np.floor(arr_size/len(neighbor_bools))), 1))

	# Then for the remaining vectors left, randomly select from neighbor_bools to fill up the array.
	if arr_size > len(bool_vecs):
		neighbor_bools = np.array(neighbor_bools)
		np.random.shuffle(neighbor_bools)
		rand_neighbor_bools = neighbor_bools[:arr_size - len(bool_vecs)].reshape(-1, boolvec_dim)
		bool_vecs = np.concatenate((bool_vecs, rand_neighbor_bools), axis=0)

	np.random.shuffle(bool_vecs)
	assert len(intersect2D(neighbor_bools, np.unique(bool_vecs, axis=0))) == len(neighbor_bools), \
					"Not all the neighbor_bools were used."

	return bool_vecs

# Testing: Convert the categorical boolean vector so that the index is also considered.
def convert_boolvec_to_position_vec(boolvec):
	assert max(boolvec) <=1 and min(boolvec) >= 0, "boolvec isn't a boolean vector."
	return [i if x == 1 else i + boolvec_dim for i, x in enumerate(boolvec, start=0)]