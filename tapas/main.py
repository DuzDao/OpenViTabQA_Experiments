import yaml
import argparse
from src.tasks.train import train_main

def main(args):
    # read config file
    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)
    
    if args.task == "train":
        train_main(config)
    else:
        raise ValueError("Only support task named 'train'!")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--task', dest="task", required=True)
    args = args_parser.parse_args()
    main(args)
    