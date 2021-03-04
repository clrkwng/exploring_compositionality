import numpy as np

# Returns a weighted sum of value times one-indexed index val.
def get_rotation_amount(bool_vec):
	return sum([i * val for i, val in enumerate(bool_vec, start=1)])

# Given classes and rotation amount, rotate class under mod num_classes.
def rotate_class(classes, rot_amts, num_classes):
	return (classes + rot_amts) % num_classes

# Return an array of random boolean vectors, where each row is a vector.
def get_rand_bool_vecs(train_size, boolvec_dim):
	return np.random.randint(2, size=(train_size, boolvec_dim))

# Returns train_size array of representative boolean vectors.
def get_rep_bool_vecs(train_size, boolvec_dim, rep_bools):
	bool_vecs = np.tile(rep_bools, ((int)(np.floor(train_size/(boolvec_dim+1))), 1))
	bool_vecs = np.concatenate((bool_vecs, np.array(rep_bools[:train_size - len(bool_vecs)]).reshape(-1, boolvec_dim)), axis=0)
	np.random.shuffle(bool_vecs)
	return bool_vecs

# Returns test_size array of all boolean vectors, dist away from train_bools
def get_dist_bool_vecs(test_size, boolvec_dim, rep_bools, dist, exclude_train_bools=True):
	neighbor_bools = get_neighbor_bools(rep_bools, boolvec_dim, dist, exclude_train_bools)

	assert len(neighbor_bools) != 0, "No neighboring boolean vectors!"
	bool_vecs = np.tile(neighbor_bools, ((int)(np.floor(test_size/len(neighbor_bools))), 1))
	bool_vecs = np.concatenate((bool_vecs, np.array(neighbor_bools[:test_size - len(bool_vecs)]).reshape(-1, boolvec_dim)), axis=0)
	np.random.shuffle(bool_vecs)
	return bool_vecs

# Returns D + 1 boolean vectors, where each vector is 1 flip away from at least one other vector.
# This provides enough boolean vectors to understand what each bit is doing for the data.
def get_representative_bools(boolvec_dim):
	bools = [tuple(np.random.randint(2, size=boolvec_dim).tolist())]
	for i in range(boolvec_dim):
		rand_bool = list(bools[np.random.randint(len(bools))]).copy()
		rand_bool[i] = 1 - rand_bool[i]
		bools.append(tuple(rand_bool))
	return bools

# Generates all bitstring vectors, with length boolvec_dim.
def get_all_bitstrings(boolvec_dim):
	n = boolvec_dim
	bitstrings = []
	for i in range(2 ** n, 2 ** (n + 1)):
		bitmask = bin(i)[3:]
		vec = tuple([int(c) for c in list(bitmask)])
		bitstrings.append(vec)
	return bitstrings

# Grabs all the bitstrings that are "dist" away from closest vec in rep_bools.
# exlude_train_bool flag will signal if we want to keep any of the vecs
# that were in our training data or not.
def get_neighbor_bools(rep_bools, boolvec_dim, dist, exclude_train_bools=True):
	all_bools = get_all_bitstrings(boolvec_dim)
	neighbors = []
	for poss_vec in all_bools:
		poss_dist = min([num_flips(poss_vec, x) for x in rep_bools])
		if poss_dist == dist:
			neighbors.append(poss_vec)
	if exclude_train_bools:
		return [list(x) for x in list(set(neighbors) - set(rep_bools))]
	else:
		return [list(x) for x in list(set(neighbors))]

# Return the minimum number of flips to get from vec1 to vec2.
# Calculates this by getting sum of absolute value of elem-wise difference.
def num_flips(vec1, vec2):
	return np.sum([np.abs(i - j) for i, j in zip(vec1, vec2)])