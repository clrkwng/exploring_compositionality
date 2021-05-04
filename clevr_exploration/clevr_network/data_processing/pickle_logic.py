import pickle

def save_pickle(obj, file):
	with open(file, 'wb') as f:
		pickle.dump(obj, f)

def load_pickle(file):
	with open(file, 'rb') as f:
		obj = pickle.load(f)
	return obj