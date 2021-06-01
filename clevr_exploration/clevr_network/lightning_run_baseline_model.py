"""
This code will run the best mini ResNet18 as a baseline on each of the distributions in 
clevr-dataset-gen/output/clevr_datasets/.
"""

from comet_ml import Experiment

import torch, os, glob

import sys
sys.path.insert(0, 'data_processing/')
from clevr_dataloader import *
from clevr_data_utils import *
sys.path.pop(0)
sys.path.insert(0, 'model/')
from lightning_clevr_lvl1_model import *
sys.path.pop(0)

comet_api_key = '5zqkkwKFbkhDgnFn7Alsby6py'
comet_workspace = 'clrkwng'
project_name = 'clevr-baseline-experiments'

RESNET18_FLAG = False
BATCH_SIZE = 256
NUM_EPOCHS = 100
OPTIMIZER = 'SGD'
LR = 0.1
MOMENTUM = 0.9
SCHEDULER = None
save_path = 'data/ignore.pt'

experiment = Experiment(
  api_key=comet_api_key,
  workspace=comet_workspace,
  project_name=project_name)

def main():
  data_dir = '../clevr-dataset-gen/output/'
  train_size = len([n for n in os.listdir(f'{data_dir}base-clevr-data3/train/images/')])
  val_size = len([n for n in os.listdir(f'{data_dir}base-clevr-data3/val/images/')])

  clevr_datasets = glob.glob('../clevr-dataset-gen/output/clevr_datasets/*')
  for dataset_path in clevr_datasets:
    val_dataset = CLEVRDataset(folder_path=f"{dataset_path}/", train_flag=False)
    # Ignore incompletely generated datasets.
    if len(val_dataset) < 12500: continue
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)

    # This grabs the concat disallowed combo.
    concat_disallowed_combo = dataset_path[dataset_path.rindex("/") + 1:]

    # Now, let's make that into a disallowed_combo_lst format.
    disallowed_combo_lst = [concat_disallowed_combo.split("_")]

    print(f"Logging {disallowed_combo_lst}.")

    model = LightningCLEVRClassifier(resnet18_flag=RESNET18_FLAG,
                                     layers=[1, 1, 1, 1],
                                     image_channels=3,
                                     batch_size=BATCH_SIZE,
                                     num_epochs=NUM_EPOCHS,
                                     train_size=train_size,
                                     val_size=val_size,
                                     out_dist_val_size=12500,
                                     optimizer=OPTIMIZER,
                                     lr=LR,
                                     momentum=MOMENTUM,
                                     scheduler=SCHEDULER,
                                     save_path=save_path,
                                     disallowed_combos_lst=disallowed_combo_lst).cuda()

    model.load_state_dict(torch.load('data/lvl1_best_model_dict.pt'))

    with experiment.test():
      model.eval()
      with torch.no_grad():
        for idx, (val_inputs, concat_labels) in enumerate(iter(val_dataloader)):
          disallowed_combo = model.concat_disallowed_combos[0]

          val_inputs = val_inputs.cuda()
          concat_preds = model.forward(val_inputs)

          attribute_set = model.map_of_attribute_sets[disallowed_combo]
          acc_map = model.val_attribute_accuracy_map[disallowed_combo]
          acc_map = update_accuracy_map(attribute_set=attribute_set,
                                        preds=torch.cat(concat_preds, dim=1),
                                        labels=concat_labels,
                                        accuracy_map=acc_map)

          if idx == model.num_out_dist_val_batches - 1:
            print(f"Logging out dist validation accuracy for {disallowed_combo} at val_batch {idx}.")

            for attribute, counts in acc_map.items():
              att_acc = round(counts[0]/counts[1], 6)
              experiment.log_metric(f"{disallowed_combo}_{attribute}_acc", att_acc)

              zero_correct, zero_total, one_correct, one_total = counts[2]
              fair_att_acc = round(0.5 * (zero_correct/zero_total + one_correct/one_total), 6)
              experiment.log_metric(f"{disallowed_combo}_fair_{attribute}_acc", fair_att_acc)

            experiment.log_parameter(f"{disallowed_combo} accuracy map", acc_map)
            acc_map.clear()

if __name__ == "__main__":
  main()
