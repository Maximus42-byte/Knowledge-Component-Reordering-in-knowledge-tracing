import argparse
from wandb_train import main

#import torch
# map the missing CUDA tensor classes to their CPU twins
#torch.cuda.LongTensor = torch.LongTensor      # type: ignore
#torch.cuda.FloatTensor = torch.FloatTensor    # type: ignore
#torch.cuda.ByteTensor  = torch.ByteTensor     # type: ignore
#torch.cuda.BoolTensor  = torch.BoolTensor     # type: ignore
import os, torch

# Pretend CUDA doesn't exist
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#torch.cuda.is_available = lambda: False     # short-circuit the test

# Create CPU stand-ins so "from torch.cuda import …" works
#torch.cuda.LongTensor  = torch.LongTensor
#torch.cuda.FloatTensor = torch.FloatTensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
