import pandas as pd
from src.dataset.preprocess import Preprocess

def preprocess_main(config):
    df_train = pd.read_csv(config["dataset"]["raw"]["train"])
    df_test = pd.read_csv(config["dataset"]["raw"]["test"])
    df_dev = pd.read_csv(config["dataset"]["raw"]["dev"])

    df_train = Preprocess(df_train, config).do_preprocess().fillna("nul")
    df_test = Preprocess(df_test, config).do_preprocess().fillna("nul")
    df_dev = Preprocess(df_dev, config).do_preprocess().fillna("nul")

    df_train.to_csv(config["dataset"]["preprocessed"]["train"])
    df_test.to_csv(config["dataset"]["preprocessed"]["test"])
    df_dev.to_csv(config["dataset"]["preprocessed"]["dev"])
