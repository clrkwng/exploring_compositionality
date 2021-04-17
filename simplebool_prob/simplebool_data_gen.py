import numpy as np
import matplotlib.pyplot as plt

# Global cache that stores the statistics of train set and each test set.
cache = {}

# Returns number of correct predictions.
def get_num_correct(labels, preds):
    preds = get_pred_class(preds)
    return (labels == preds).float().sum()

# g : R^2 -> {0,1}.
# Here, each x in X is in R^2.
def true_g(X):
	return np.where(np.sum(X, axis=1) < 0.5, 0, 1)

# Given g, and each x in X is in R^3.
def true_f(g, X):
	g_result = g(X[:, :2])
	return (g_result) * (1 - X[:, 2]) + (1 - g_result) * X[:, 2]

# Standardize the data (subtract mean, divide by std).
# Will use training data's mean and std for both train and test data.
def standardize_data(X_orig, train_mode=False):
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

# Get the class from the logits, with the boundary being 0 for the classes.
def get_pred_class(preds):
    return (preds > 0).float()

# Returns whether val \in [lowval, highval].
def is_in(val, lowval, highval):
	return lowval <= val <= highval

# Save plot, specifically used during training phase.
def save_plt(X_orig, y_vals, preds, epoch, acc):
		acc = np.round(acc, 6)    
		preds = get_pred_class(preds)

		zero_indices = (X_orig[:,2] == 0)
		one_indices = (X_orig[:,2] == 1)

		# Plotting the boolean = 0 case first.
		X_orig_zero = X_orig[zero_indices]
		y_vals_zero = y_vals[zero_indices]
		preds_zero = preds[zero_indices]
		fig = plt.figure(figsize=(12,12))
		ax = plt.axes(projection='3d')
		ax.scatter3D(X_orig_zero[:,0], X_orig_zero[:,1], y_vals_zero, c="#143D59")
		ax.scatter3D(X_orig_zero[:,0], X_orig_zero[:,1], preds_zero, c="#F4B41A")
		ax.set_xlabel(f"Epoch: {epoch}")
		ax.set_ylabel(f"Boolean: {0}")
		plt.savefig(f"model_guesses_over_epoch/boolean0/plot{epoch}_{0}.png", bbox_inches="tight")
		plt.close()

		# Plotting the boolean = 1 case first.
		X_orig_one = X_orig[one_indices]
		y_vals_one = y_vals[one_indices]
		preds_one = preds[one_indices]
		fig = plt.figure(figsize=(12,12))
		ax = plt.axes(projection='3d')
		ax.scatter3D(X_orig_one[:,0], X_orig_one[:,1], y_vals_one, c="#143D59")
		ax.scatter3D(X_orig_one[:,0], X_orig_one[:,1], preds_one, c="#F4B41A")
		ax.set_xlabel(f"Epoch: {epoch}")
		ax.set_ylabel(f"Boolean: {1}")
		plt.savefig(f"model_guesses_over_epoch/boolean1/plot{epoch}_{1}.png", bbox_inches="tight")
		plt.close()
		

# Number of training points will be split_sizes[0] + split_sizes[1].
def get_train_data(split_sizes):
	global cache

	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_2 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_2), axis=1)
	y1 = true_f(true_g, X1)

	X2_01_1 = np.random.uniform(0, 0.3, (n1//2, 2))
	X2_01_2 = np.random.uniform(4.7, 5, (n1//2, 2))
	X2_01 = np.concatenate((X2_01_1, X2_01_2), axis=0)
	X2_2 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_2), axis=1)
	y2 = true_f(true_g, X2)

	X_train = np.concatenate((X1, X2), axis=0)
	y_train = np.concatenate((y1, y2), axis=0)

	X, X_mean, X_std = standardize_data(X_train, train_mode=True)
	cache["X_train_mean"] = X_mean
	cache["X_train_std"] = X_std

	return X, y_train
	
# This is a test of compositionality (on data that model hasn't been trained on).
def get_test_splitA(test_size):
	global cache

	X_01 = np.random.uniform(0, 5, (test_size, 2))
	X_02 = np.ones((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testA_mean"] = X_mean
	cache["X_testA_std"] = X_std

	return X, X_test, y_test

# This test matches training data distribution.
def get_test_splitB(split_sizes):
	global cache

	n0 = split_sizes[0]
	n1 = split_sizes[1]

	X1_01 = np.random.uniform(0, 5, (n0, 2))
	X1_02 = np.zeros((n0, 1))
	X1 = np.concatenate((X1_01, X1_02), axis=1)

	X2_01 = np.random.uniform(0, 1, (n1, 2))
	X2_02 = np.ones((n1, 1))
	X2 = np.concatenate((X2_01, X2_02), axis=1)

	X_test = np.concatenate((X1, X2), axis=0)
	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testB_mean"] = X_mean
	cache["X_testB_std"] = X_std

	return X, X_test, y_test

# This test looks at x in (0,1), which is overlapped region for boolean = 1 and boolean = 0.
def get_test_splitC(test_size):
	global cache

	X_01 = np.random.uniform(0, 1, (test_size, 2))
	X_02 = np.zeros((test_size, 1))
	X_test = np.concatenate((X_01, X_02), axis=1)

	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testC_mean"] = X_mean
	cache["X_testC_std"] = X_std

	return X, X_test, y_test

# This test looks at x in (-10,10), which is used for plotting during training.
def get_test_splitD(test_size):
	global cache

	X1_01 = np.random.uniform(-10, 10, (test_size, 2))
	X1_02 = np.zeros((test_size, 1))
	X1 = np.concatenate((X1_01, X1_02), axis=1)

	X2_01 = np.random.uniform(-10, 10, (test_size, 2))
	X2_02 = np.ones((test_size, 1))
	X2 = np.concatenate((X2_01, X2_02), axis=1)

	X_test = np.concatenate((X1, X2), axis=0)
	y_test = true_f(true_g, X_test)

	X, X_mean, X_std = standardize_data(X_test)
	cache["X_testD_mean"] = X_mean
	cache["X_testD_std"] = X_std

	return X, X_test, y_test