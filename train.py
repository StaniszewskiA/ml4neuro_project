import os
import sys
import warnings
from argparse import ArgumentParser
import torch
import torch_geometric
import wandb
import numpy as np
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from models import GATv2Lightning
from utils.dataloader_utils import (
    GraphDataset,
    HDFDataset_Writer,
    HDFDatasetLoader,
    save_data_list,
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # DISABLED ON PURPOSE
torch_geometric.seed_everything(42)
api_key_file = open("wandb_api_key.txt", "r")
API_KEY = api_key_file.read()

api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = ArgumentParser()
    parser.add_argument("--timestep", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--exp_name", type=str, default="eeg_exp")
    args = parser.parse_args()
else:
    class Args: # Default args for Jupyter
        timestep = 6
        epochs = 25
        batch_size = 256
        n_splits = 5
        cache_dir = "data/cache"
        exp_name = "eeg_exp"

    args = Args()

CONFIG = {
    "timestep": args.timestep,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
}



def kfold_cval():
    writer = HDFDataset_Writer(sample_timestep=args.timestep, cache_folder=args.cache_dir)
    cache_file_path = writer.get_dataset()
    loader = HDFDatasetLoader(root=cache_file_path)
    full_dataset = GraphDataset(loader.get_datasets()[0])

    kfold = MultilabelStratifiedKFold(n_splits=args.n_splits, random_state=42, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(full_dataset)), [data.y for data in full_dataset])):
        print(f"Fold {fold}")
        wandb.init(project="eeg_project", group=args.exp_name, name=f"fold_{fold}", config=CONFIG)

        train_data = [full_dataset[idx].to(device) for idx in train_idx]
        test_data = [full_dataset[idx].to(device) for idx in test_idx]
        save_data_list(test_data, f"saved_folds/{args.exp_name}/fold_{fold}/data.pt")

        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        model = GATv2Lightning(features_shape=train_data[0].x.shape[-1]).to(device)
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            logger=pl.loggers.WandbLogger(),
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        trainer.test(model, valid_dataloader)
        wandb.finish()