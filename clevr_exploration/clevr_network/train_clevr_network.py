from comet_ml import Experiment

import sys
sys.path.insert(0, 'model/')
from model_v5 import *
sys.path.pop(0)
sys.path.insert(0, 'data_processing/')
from clevr_dataset import *
sys.path.pop(0)

import torch.optim as optim
import math

BATCH_SIZE = 256
NUM_BATCHES = math.ceil(1.0 * 10000 / BATCH_SIZE)
# LR = 1e-2
MOMENTUM = 0.9

experiment = Experiment(api_key='5zqkkwKFbkhDgnFn7Alsby6py', project_name='clevr-network', workspace='clrkwng')

def train(model, criterion, optimizer, train_loader, valid_loader, save_model_path, num_epochs):
	epochs_no_improve = 0
	valid_loss_min = np.Inf
	valid_max_acc = 0
	history = []

	t = tqdm(range(1, num_epochs + 1), miniters=100)
	best_val_acc = 0

	count = 0
	with experiment.train():
		step = 0
		for epoch in t:
			total_loss = 0
			cube_correct = 0
			cylinder_correct = 0
			sphere_correct = 0
			train_total = 0

			for i, (inputs, labels) in enumerate(train_loader, 0):
				model.train()
				inputs = inputs.cuda()
				cube_labels, cylinder_labels, sphere_labels = labels[:,0].cuda(), labels[:,1].cuda(), labels[:,2].cuda()

				# Forward + Backward + Optimize
				optimizer.zero_grad()
				cube_preds, cylinder_preds, sphere_preds = model(inputs)
				cube_loss = criterion(cube_preds, cube_labels)
				cylinder_loss = criterion(cylinder_preds, cylinder_labels)
				sphere_loss = criterion(sphere_preds, sphere_labels)
				loss = cube_loss + cylinder_loss + sphere_loss
				loss.backward()
				optimizer.step()

				cube_correct += single_label_get_num_correct(cube_preds, cube_labels)
				cylinder_correct += single_label_get_num_correct(cylinder_preds, cylinder_labels)
				sphere_correct += single_label_get_num_correct(sphere_preds, sphere_labels)
				train_total += labels.size(0)
				total_loss += loss.item()

				step += 1
				if step % 5 == 0:
					count += 1
					cube_acc = round(cube_correct/train_total, 6)
					cylinder_acc = round(cylinder_correct/train_total, 6)
					sphere_acc = round(sphere_correct/train_total, 6)
					experiment.log_metric("cube_acc", cube_acc, step=step)
					experiment.log_metric("cylinder_acc", cylinder_acc, step=step)
					experiment.log_metric("sphere_acc", sphere_acc, step=step)

					# Getting validation accuracy now.
					with experiment.validate():
						model.eval()
						val_cube_correct = 0
						val_cylinder_correct = 0
						val_sphere_correct = 0
						with torch.no_grad():
							for val_inputs, val_labels in valid_loader:
								val_inputs = val_inputs.cuda()
								val_cube_labels, val_cylinder_labels, val_sphere_labels = val_labels[:,0].cuda(), val_labels[:,1].cuda(), val_labels[:,2].cuda()
								val_cube_preds, val_cylinder_preds, val_sphere_preds = model(val_inputs)

								val_cube_correct += single_label_get_num_correct(val_cube_preds, val_cube_labels)
								val_cylinder_correct += single_label_get_num_correct(val_cylinder_preds, val_cylinder_labels)
								val_sphere_correct += single_label_get_num_correct(val_sphere_preds, val_sphere_labels)

							val_cube_acc = round(val_cube_correct/len(valid_loader), 6)
							val_cylinder_acc = round(val_cylinder_correct/len(valid_loader), 6)
							val_sphere_acc = round(val_sphere_correct/len(valid_loader), 6)

							experiment.log_metric("cube_acc", val_cube_acc, step=step)
							experiment.log_metric("cylinder_acc", val_cylinder_acc, step=step)
							experiment.log_metric("sphere_acc", val_sphere_acc, step=step)

							if (val_cube_acc + val_cylinder_acc + val_sphere_acc) / 3 > best_val_acc:
								best_val_acc = (val_cube_acc + val_cylinder_acc + val_sphere_acc) / 3
								torch.save(model.state_dict(), save_model_path)

			epoch_loss = round(total_loss/NUM_BATCHES, 6)
			t.set_description(f"Epoch: {epoch}/{num_epochs}, Loss: {epoch_loss}, Cube acc: {cube_acc},\
													Cylinder acc: {cylinder_acc}, Sphere acc: {sphere_acc}")


def main():
	model = MiniResNet18().cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

	train_dataset = CLEVRDataset('../clevr-dataset-gen/output/train/')
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

	valid_dataset = CLEVRDataset('../clevr-dataset-gen/output/val/')
	valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

	save_model_path = 'data/clevr_model_state_dict.pt'
	num_epochs = 5000

	train(model, criterion, optimizer, train_loader, valid_loader, save_model_path, num_epochs)

if __name__ == "__main__":
	main()