from loguru import logger
import yaml
import argparse
from tqdm import tqdm

from src.tasks.train import train_main
from src.tasks.preprocess_data import preprocess_main

LOG_LEVEL = "DEBUG"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
LOG_COLORIZE = True
LOG_BACKTRACK = True
LOG_DIAGNOSE = True


def main(args):
    # set up logger
    logger.add(args.log_file,
               level=LOG_LEVEL,
               format=LOG_FORMAT,
               colorize=LOG_COLORIZE,
               backtrace=LOG_BACKTRACK,
               diagnose=LOG_DIAGNOSE)
    logger.add(tqdm.write)

    # read config file
    with open(args.conf) as conf_file:
        config = yaml.safe_load(conf_file)

    # do task
    if args.task == "train":
        train_main(config, logger)
    elif args.task == "preprocess":
        preprocess_main(config)
    else:
        raise ValueError("Only support task named 'train'!")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--config', dest='conf', required=True,
        help = 'The path of config file (.yaml file)')

    args_parser.add_argument(
        '--task', dest='task', required=True,
        help = """We have 2 task available: 
                    - Training task: 'train'
                    - Inference task: 'predict' | 'inference'
                    - Get preprocesses data: 'preprocess'""")
    
    args_parser.add_argument(
        '--log', dest='log_file', default='/content/drive/MyDrive/tableqa/logs/info.log',
        help = """Provide path of logging file.
                  Default = "/content/drive/MyDrive/tableqa/logs/info.log"
               """)
    
    args = args_parser.parse_args()
    main(args)
