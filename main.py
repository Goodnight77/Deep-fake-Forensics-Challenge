from model import build_resnet50_unet
from unet_models import Models
from utils_data import Utils
import argparse

if __name__ == "__main__" :
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=50,
                      help="number of epochs")
    parser.add_argument('--n_bs', type=int, default=32,
                      help="the batch_size")
    parser.add_argument('--train_dataframe', type=str, default="/path/to/train_data.csv",
                      help="train_data")
    parser.add_argument('--val_dataframe', type=str, default="/path/to/val_data.csv",
                      help="val_data")
    #parser.add_argument('--mask_p', type=str, default="/output filled data/",
                      #help="mask_path")

    # parser.add_argument('--N_data', type=int,
    #                   help="Data number in the dataframe")
    args = parser.parse_args()
    try1=Models(args.n_epoch,args.n_bs,args.Input_p,args.mask_p)

    try1.id_model(args.n_unet)
    try1.compile_model()
    try1.train()
