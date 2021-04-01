from bitstring_utils import *
import math
import torch
import torch.optim as optim

import sys
from tqdm import tqdm
sys.path.insert(0, '../models/')
from bitstr_model import *

num_epochs = 1000
lr = 1e-2
batch_size = 1024
emb_dims = [num_symbols * boolvec_dim, 2 * num_symbols * boolvec_dim]
lin_layer_sizes = [128, 512, 128, 32, 8]
output_size = 1
hidden_drop_p = 0
batch_flag = True

hyper_params = {
	"num_epochs": num_epochs,
	"lr": lr,
	"batch_size": batch_size,
	"emb_dims": emb_dims,
	"num_symbols": num_symbols,
	"lin_layer_sizes": lin_layer_sizes,
	"output_size": output_size,
	"hidden_drop_p": hidden_drop_p,
	"batch_flag": batch_flag
}

experiment_bools = {
	"train_bools": train_bools,
	"test_bools": test_bools
}

print(f"train_bools: {train_bools}")
print(f"test_bools: {test_bools}")

with open('../ssh_keys/comet_api_key.txt', 'r') as file:
	comet_key = file.read().replace('\n', '')

experiment = Experiment(api_key=comet_key, project_name="bitstr_model", workspace="clrkwng")
experiment.log_parameters(hyper_params)
experiment.log_parameters(experiment_bools)

# Test the model on a single boolean vector.
# Note: The boolean vector input hasn't been put through 'convert_boolvec_to_position_vec' yet.
def test_model(bool_vec):
	bool_vec = torch.tensor(convert_boolvec_to_position_vec(bool_vec)).long().cuda()
	bool_vec = bool_vec.reshape((1, 10))
	model = BitNet(boolvec_dim=boolvec_dim, emb_dims=emb_dims, lin_layer_sizes=lin_layer_sizes, \
								 output_size=output_size, hidden_drop_p=hidden_drop_p, batch_flag=batch_flag).cuda()
	save_path = "../saved_model_params/bitstr_model_state_dict.pt"
	model.load_state_dict(torch.load(save_path))
	
	model.eval()
	return model(bool_vec).item()

# Takes in train_size, and val_size. Returns train, validation split.
def get_train_valid_data(train_size, val_size):
	X_train, y_train = get_train_data(train_size)
	X_valid, y_valid = get_train_data(val_size)

	X_train = torch.tensor(X_train).long().cuda()
	y_train = torch.tensor(y_train).long().cuda()

	train_data = []
	for i in range(len(X_train)):
		train_data.append([X_train[i], y_train[i]])

	X_valid = torch.tensor(X_valid).long().cuda()
	y_valid = torch.tensor(y_valid).long().cuda()

	return train_data, (X_valid, y_valid)

def train_model(model, trainloader, valid_data, num_batches, train_size, save_path):
	test_size = 1000
	X_test, y_test = get_test_data(test_size)
	X_test = torch.tensor(X_test).long().cuda()
	y_test = torch.tensor(y_test).long().cuda()

	t = tqdm(range(1, num_epochs + 1), miniters=100)
	best_val_loss = 999999999

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	with experiment.train():
		step = 0
		for epoch in t:
			total_loss = 0
			for i, (inputs, labels) in enumerate(trainloader, 0):
				model.train()
				
				# Forward + Backward + Optimize
				optimizer.zero_grad()
				preds = model(inputs).reshape(labels.shape)
				loss = criterion(labels.float(), preds.float())
				loss.backward()
				optimizer.step()

				total_loss += loss.item()

				step += 1
				if step % 5 == 0:
					# Getting the validation loss now.
					with experiment.validate():
						model.eval()
						with torch.no_grad():
							val_inputs, val_labels = valid_data
							val_preds = model(val_inputs).reshape(val_labels.shape)
							val_loss = criterion(val_labels.float(), val_preds.float()).item()
							experiment.log_metric("loss", val_loss, step=step)

							if val_loss < best_val_loss:
								best_val_loss = val_loss
								torch.save(model.state_dict(), save_path)
					
					with experiment.test():
						model.eval()
						with torch.no_grad():
							test_preds = model(X_test)
							test_loss = criterion(y_test.float(), test_preds.float()).item()
							experiment.log_metric("loss", test_loss, step=step)

			epoch_loss = round(total_loss/num_batches, 6)
			t.set_description(f"Epoch: {epoch}/{num_epochs}, Loss: {epoch_loss}")

def main():
		assert torch.cuda.is_available(), "GPU isn't available."

		train_size, valid_size = bool_train_num, bool_train_num
		train_data, valid_data = get_train_valid_data(train_size, valid_size)

		trainloader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
		num_batches = math.ceil(1.0 * train_size / batch_size)
		
		model = BitNet(boolvec_dim=boolvec_dim, emb_dims=emb_dims, lin_layer_sizes=lin_layer_sizes, \
									 output_size=output_size, hidden_drop_p=hidden_drop_p, batch_flag=batch_flag).cuda()

		save_path = "../saved_model_params/bitstr_model_state_dict.pt"
		train_model(model, trainloader, valid_data, num_batches, train_size, save_path)

if __name__ == "__main__":
	main()
				