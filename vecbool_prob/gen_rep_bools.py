from bool_utils import *
import sys
sys.path[0] = '../pickled_files/'
from pickle_logic import *

boolvec_dim = 10

def gen_bools():
	rep_bools = get_representative_bools(boolvec_dim)
	save_pickle(rep_bools, "../pickled_files/rep_bools.pickle")
	return rep_bools
