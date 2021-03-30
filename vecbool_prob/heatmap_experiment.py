from comet_ml import Experiment
import torch

import sys
from vecbool_data_gen import *
sys.path.insert(0, '../models/')
from model_v2 import *

# Generate the continuous data, with uniform number of each ground truth class.
def get_cont_data(cont_size):
	X = np.random.uniform(hyper_params["cont_range"][0], hyper_params["cont_range"][1], size=(cont_size, hyper_params["num_cont"]))
	X, true_labels = true_g(X)
	X = standardize_data(X)
	
	return X, true_labels

# Given all the cont_data generated and a single bool_vec, return the predictions from the model.
def eval_model(cont_data, bool_vec, model, y_true):
	cont_x = torch.tensor(cont_data).cuda().float()

	if convertBooleanFlag:
		bool_vec = convert_boolvec_to_position_vec(bool_vec)

	cat_x = get_rep_bool_vecs(len(cont_x), hyper_params["boolvec_dim"], [bool_vec])
	cat_x = torch.tensor(cat_x).cuda().long()
	y_true = torch.tensor(y_true).cuda()

	model.eval()
	with torch.no_grad():
		preds = model(cont_x, cat_x)
		test_acc = get_num_correct(preds, y_true, k=0) / len(y_true)
		preds = preds.max(1, keepdim=True)[1]

	return tensor_to_numpy(preds).reshape((-1,)), test_acc

# Get the true offsets, one in mod space and the other in unmodded space.
def calc_true_offsets(true_labels_B_y, true_labels_B_x):
	poss_offsets = true_labels_B_y - true_labels_B_x
	return [int(x) for x in np.unique(poss_offsets)]

# Generate the heatmaps, given the already generated continuous data, model, and true_labels.
def gen_heatmap(cont_data, model, true_labels):
	for B_y in hyper_params["neighbor_bools"]:
		x_vals = []
		y_vals = []

		# correct_map has B_x index to offset_correct_pct mapping.
		# true_offset_map has B_x index to tuple of true_offsets mapping.
		correct_map = {}
		true_offset_map = {}
		for i, B_x in enumerate(hyper_params["rep_bools"]):
			true_labels_B_y = rotate_class(true_labels, np.tile(get_rotation_amount(B_y), (len(true_labels),)), hyper_params["num_classes"])
			y_hat, y_hat_acc = eval_model(cont_data, B_y, model, true_labels_B_y)

			true_labels_B_x = rotate_class(true_labels, np.tile(get_rotation_amount(B_x), (len(true_labels),)), hyper_params["num_classes"])
			y, y_acc = eval_model(cont_data, B_x, model, true_labels_B_x)
			
			offsets = y_hat - y

			# Calculating the percentage of correct offset calculations.
			unique_counts = np.unique(offsets, return_counts=True)
			true_offsets = calc_true_offsets(true_labels_B_y, true_labels_B_x)
			offset_correct_total = 0
			for val, count in zip(unique_counts[0], unique_counts[1]):
				if val in true_offsets:
					offset_correct_total += count
			offset_correct_pct = (1.0 * offset_correct_total) / len(cont_data) * 100

			correct_map[i] = np.round(offset_correct_pct, 3)
			true_offset_map[i] = tuple(true_offsets)

			x_vals.extend([i] * len(offsets))
			y_vals.extend(offsets.reshape((-1,)).tolist())
		
		heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=[5,19])
		extent = [xedges[0], xedges[-1], yedges[0], 10]

		plt.clf()
		fig, ax = plt.subplots()
		plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=plt.cm.Blues)

		# Setting up the legend for percentage correct, as well as the true offsets.
		legend = ""
		for i, B_x in enumerate(hyper_params["rep_bools"]):
			legend += convert_boolvec_to_str(B_x) + ": " + str(correct_map[i]) + "%, " + str(true_offset_map[i])
			legend += "\n"
		plt.text(-10, -2, legend)

		plt.xticks(list(range(len(hyper_params["rep_bools"]))), \
			[convert_boolvec_to_str(vec) for vec in hyper_params["rep_bools"]])
		plt.yticks(list(range(-9, 11)), list(range(-9, 11)))
		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
		plt.title(f"Heatmap for: {convert_boolvec_to_str(B_y)}")
		plt.tight_layout()
		plt.savefig(f"heatmaps/heatmap{convert_boolvec_to_str(B_y)}.png")

def main():
	model = WangNet(boolvec_dim=hyper_params["boolvec_dim"], emb_dims=hyper_params["emb_dims"], num_cont=hyper_params["num_cont"], \
		lin_layer_sizes=hyper_params["lin_layer_sizes"], output_size=hyper_params["num_classes"], hidden_drop_p=hyper_params["hidden_drop_p"], \
		batch_flag=hyper_params["batch_flag"]).cuda()
	save_path = "../saved_model_params/vecbool_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))

	cont_data, true_labels = get_cont_data(10000)

	gen_heatmap(cont_data, model, true_labels)

if __name__ == "__main__":
	main()