# encoding utf-8
import numpy as np
import pandas as pd
from utils.utils import Z_Score
from utils.utils import generate_dataset
from utils.utils import generate_asist_dataset


def Data_load(config, timesteps_input, timesteps_output):
    W_nodes =np.load("./data/TRI/T.npy").astype(np.float32)
    X = pd.read_csv(config['V_nodes'], header=None).to_numpy(np.float32) * 100
    V_confirmed = pd.read_csv(config['V_confirmed'], header=None).to_numpy(np.float32)
    V_cured = pd.read_csv(config['V_cured'], header=None).to_numpy(np.float32)
    V_suspected = pd.read_csv(config['V_suspected'], header=None).to_numpy(np.float32)
    V_dead = pd.read_csv(config['V_dead'], header=None).to_numpy(np.float32)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    X, X_mean, X_std = Z_Score(X)

    V_confirmed = np.reshape(V_confirmed, (V_confirmed.shape[0], V_confirmed.shape[1], 1)).transpose((1, 2, 0))
    V_confirmed, _, _ = Z_Score(V_confirmed)
    V_cured = np.reshape(V_cured, (V_cured.shape[0], V_cured.shape[1], 1)).transpose((1, 2, 0))
    V_cured, _, _ = Z_Score(V_cured)
    V_suspected = np.reshape(V_suspected, (V_suspected.shape[0], V_suspected.shape[1], 1)).transpose((1, 2, 0))
    V_suspected, _, _ = Z_Score(V_suspected)
    V_dead = np.reshape(V_dead, (V_dead.shape[0], V_dead.shape[1], 1)).transpose((1, 2, 0))
    V_dead, _, _ = Z_Score(V_dead)
    V_conbine = np.concatenate((V_confirmed, V_cured, V_suspected, V_dead), axis=1)

    index_1 = int(X.shape[2] * 0.8)
    index_2 = int(X.shape[2])

    train_original_data = X[:, :, :index_2]
    train_asist = V_conbine[:, :, :index_2]
    val_original_data = X[:, :, :index_2]
    val_asist = V_conbine[:, :, :index_2]


    train_input, train_target = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output)
    train_asist_dataset = generate_asist_dataset(train_asist, timesteps_input, timesteps_output)
    val_asist_dataset = generate_asist_dataset(val_asist, timesteps_input, timesteps_output)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set[
        'eval_target'], data_set["train_asist"], data_set["eval_asist"], data_set['X_mean'], data_set['X_std'], \
        = train_input, train_target, evaluate_input, evaluate_target, train_asist_dataset, val_asist_dataset, X_mean, X_std

    return W_nodes, data_set

