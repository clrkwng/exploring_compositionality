from comet_ml import Experiment
import numpy as np
from itertools import permutations

# Denotes how many symbols in the vector (if 2, this is just a normal bitstring).
num_symbols = 2
boolvec_dim = 10

bool_train_num = (num_symbols**boolvec_dim) // 2

# Generates all bitstring vectors, with length boolvec_dim.
def get_all_bitstrings(boolvec_dim):
	n = boolvec_dim
	bitstrings = []
	for i in range(2 ** n, 2 ** (n + 1)):
		bitmask = bin(i)[3:]
		vec = tuple([int(c) for c in list(bitmask)])
		bitstrings.append(vec)

	assert len(np.unique(bitstrings, axis=0)) == np.power(num_symbols, boolvec_dim), \
					"Not enough bitstrings."
	return bitstrings

# Returns the decimal interpretation of the boolean vector.
# The base used is num_symbols.
def bool_to_dec(bool_vec):
	dec_val = sum([(num_symbols**i) * val for i, val in enumerate(bool_vec)])
	return dec_val

# Testing: Convert the categorical boolean vector so that the index is also considered.
def convert_boolvec_to_position_vec(boolvec):
	assert max(boolvec) < num_symbols and min(boolvec) >= 0, "boolvec isn't properly initialized vector."
	assert len(boolvec) == boolvec_dim, "boolvec is wrong length."
	return [i + boolvec_dim * x for i, x in enumerate(boolvec, start=0)]

# Returns row intersection between arr1 and arr2
def intersect2D(arr1, arr2):
	return np.array([x for x in set(tuple(x) for x in arr1) & set(tuple(x) for x in arr2)])

# Samples train_num of all the booleans for training, the rest are for test.
def get_train_test_split(bool_train_num, boolvec_dim):
	assert bool_train_num < num_symbols**boolvec_dim, "Not enough boolean vectors altogether."

	all_bools = np.asarray(get_all_bitstrings(boolvec_dim))
	all_indices = list(range(len(all_bools)))
	train_indices = np.random.choice(all_indices, bool_train_num, replace=False)
	test_indices = np.asarray(list(set(all_indices) - set(train_indices)))

	return all_bools[train_indices], all_bools[test_indices]

train_bools, test_bools = get_train_test_split(bool_train_num, boolvec_dim)

# Returns arr_size array of representative boolean vectors.
def get_rep_bool_vecs(arr_size, boolvec_dim, rep_bools):
	assert len(rep_bools) <= arr_size, "Desired array size is less than the number of rep_bools."
	# assert rep_bools_len == len(rep_bools), "Mismatch between boolvec_dim and dim of rep_bools."

	# This tries to fit as many multiples of rep_bools into arr_size.
	bool_vecs = np.tile(rep_bools, ((int)(np.floor(arr_size/(len(rep_bools)))), 1))

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

# Returns X_train, y_train
def get_train_data(train_size):
	X_train = get_rep_bool_vecs(train_size, boolvec_dim, train_bools)
	y_train = bool_to_dec(X_train.T)
	
	for i in range(len(X_train)):
		X_train[i] = convert_boolvec_to_position_vec(X_train[i])

	return np.array(X_train), np.array(y_train)
	
# Returns X_test, y_test
def get_test_data(test_size):
	X_test = get_rep_bool_vecs(test_size, boolvec_dim, test_bools)
	y_test = bool_to_dec(X_test.T)

	for i in range(len(X_test)):
		X_test[i] = convert_boolvec_to_position_vec(X_test[i])

	return np.array(X_test), np.array(y_test)

