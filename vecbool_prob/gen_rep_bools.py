from bool_utils import *
import sys
sys.path[0] = '../pickled_files/'
from pickle_logic import *

# Used to instantiate and store the rep_bools, which will be used in training.
def gen_bools():

	# Change this line if want to hardcode the rep_bools in for a specific example.
	# rep_bools = [(0,1,1,1,0), (1,1,1,1,0), (0,0,1,1,0), (0,1,0,1,0), (0,0,1,0,0), (0,0,1,1,1)]
	rep_bools = get_representative_bools(boolvec_dim)
	if rep_bools_len != boolvec_dim + 1:
		save_pickle(rep_bools, "../pickled_files/" + str(rep_bools_len) + "_rep_bools.pickle")
	else:
		save_pickle(rep_bools, "../pickled_files/rep_bools.pickle")
	return rep_bools
