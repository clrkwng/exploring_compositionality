import numpy as np

# Global cache that stores the statistics of train set and each test set.
cache = {}

num_classes = 10
boolvec_dim = 4
cont_range = [0, 5]

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, train_mode=False):
	global cache

	X = np.copy(X_orig)

	if train_mode:
		X_mean = np.mean(X[:, :2], axis=0)
		X_std = np.std(X[:, :2], axis=0)
		X[:, :2] -= X_mean
		X[:, :2] /= X_std

	else:
		assert "X_train_mean" in cache and "X_train_std" in cache,\
			"Train data statistics have not been cached yet."

		X_mean = np.mean(X[:, :2], axis=0)
		X_std = np.std(X[:, :2], axis=0)
		X[:, :2] -= cache["X_train_mean"]
		X[:, :2] /= cache["X_train_std"]

	return X, X_mean, X_std

# Return unstandardized data, used for plotting.
def unstandardize_data(X_orig, X_mean, X_std):
	X = np.copy(X_orig)

	X[:, :2] *= X_std
	X[:, :2] += X_mean

	return X

# g : R^2 -> {0, 1, ..., num_classes - 1}.
# Here, each x in X is in R^2.
def true_g(X):
	X_sum = np.sum(X, axis=1)

	# Percentage gets what percentage the sum of x1, x2 is of the the possible continuous sum.
	percentage = (X_sum - (2 * cont_range[0])) / (2 * (cont_range[1] - cont_range[0]))
	return np.floor(percentage * num_classes)

# Given g, and each x in X is in R^3.
def true_f(g, X):
	true_labels = g(X[:, :2])
	rot_amts = get_rotation_amount(X[:, 2:].T)
	rotated_labels = rotate_class(true_labels, rot_amts)
	return true_labels, rotated_labels

# Returns a weighted sum of value times one-indexed index val.
def get_rotation_amount(bool_vec):
	return sum([i * val for i, val in enumerate(bool_vec, start=1)])

# Given classes and rotation amount, rotate class under mod num_classes.
def rotate_class(classes, rot_amts):
	return (classes + rot_amts) % num_classes

# Return an array of random boolean vectors, where each row is a vector.
def get_bool_vecs(train_size):
	return np.random.randint(2, size=(train_size, boolvec_dim))

def get_train_data(train_size):
	X_01 = np.random.uniform(cont_range[0], cont_range[1], size=(train_size, 2))
	X_02 = get_bool_vecs(train_size)
	X = np.concatenate((X_01, X_02), axis=1)
	true_labels, rotated_labels = true_f(true_g, X)

	X, X_mean, X_std = standardize_data(X, train_mode=True)
	cache["X_train_mean"] = X_mean
	cache["X_train_std"] = X_std

	return X, true_labels, rotated_labels

# def get_test_splitB(test_size):
