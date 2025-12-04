import numpy as np
import pandas as pd
from Dataset.dataset import Flchain, SimLinearPH, SimNonLinearPH3, SEERBreastCancer
from Models.train_helper import coxse_cv, coxsenam_cv, coxnam_cv, deepsurv_cv, cph_cv
from Utils.utils import configure_logger
import random

# region hp search functions
def random_search_deepsurv(ds, k_runs=1, df=None, no_change=10, exp_name='surv', logdir='logs', exp_id='exp', batch_size=2000, epochs=1000000, patience=100, early_stopping=True):
    logger = configure_logger(exp_name, logdir)
    if df is None:
        df = pd.DataFrame(columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'c_index'])

    selected = []
    # initial values
    num_layers=4
    num_nodes=16
    act='relu'
    l2w=0.01
    dropoutp=0.1
    learning_rate=0.001

    best_c_index = 0 if df['c_index'].max() is np.nan else df['c_index'].max() #0

    best_num_layers = num_layers
    best_num_nodes = num_nodes
    best_act = act
    best_l2w = l2w
    best_dropoutp = dropoutp
    best_learning_rate = learning_rate

    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 1000)):
        i += 1
        logger.info(
            f'{i} - Testing num_layers: {num_layers}, num_nodes: {num_nodes}, act: {act}, l2w: {l2w}, dropoutp: {dropoutp}, learning_rate: {learning_rate}')

        if (num_layers, num_nodes, act, l2w, dropoutp, learning_rate) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((num_layers, num_nodes, act, l2w, dropoutp, learning_rate))
            counter += 1

            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                                batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                                verbose=False, exp_id=exp_id)

            logger.info(f'{str(c_index_list)}, avg: {c_index}')
            df.loc[len(df)] = [i, num_layers, num_nodes, act, l2w, dropoutp, learning_rate, c_index]
            df.to_csv(f'{logdir}/{exp_name}_results.csv', index=False)
            if (c_index > best_c_index):
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_num_layers = num_layers
                best_num_nodes = num_nodes
                best_act = act
                best_l2w = l2w
                best_dropoutp = dropoutp
                best_learning_rate = learning_rate
        else:
            logger.info('Already Selected')
            num_selected += 1
        num_layers = random.choice([1, 2, 3, 4])
        num_nodes = random.choice([4, 8, 16, 32, 64, 128])
        act = 'relu'
        l2w = random.choice([0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1])
        dropoutp = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        learning_rate = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(4,1,-1, dtype=float))]))

    return best_num_layers, best_num_nodes, best_act, best_l2w, best_dropoutp, best_learning_rate, best_i, best_c_index


def random_search_coxse(ds, k_runs=1, df=None, no_change=10, exp_name='coxse', logdir='logs', exp_id='exp', batch_size=2000, epochs=1000000, patience=100, early_stopping=True):
    logger = configure_logger(exp_name, logdir)
    if df is None:
        df = pd.DataFrame(
            columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'alpha', 'beta',
                     'c_index'])

    selected = []
    # initial values
    num_layers = 4
    num_nodes = 16
    act = 'relu'
    l2w = 0.01
    dropoutp = 0.1
    learning_rate = 0.001
    alpha = 0.1
    beta = 0

    best_c_index = 0 if df['c_index'].max() is np.nan else df['c_index'].max() #0

    best_num_layers = num_layers
    best_num_nodes = num_nodes
    best_act = act
    best_l2w = l2w
    best_dropoutp = dropoutp
    best_learning_rate = learning_rate
    best_alpha = alpha
    best_beta = beta

    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 1000)):
        i += 1
        logger.info(
            f'{i} - Testing num_layers: {num_layers}, num_nodes: {num_nodes}, act: {act}, l2w: {l2w}, dropoutp: {dropoutp}, learning_rate: {learning_rate}, alpha: {alpha}, beta: {beta}')

        if (num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta))
            counter += 1

            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=alpha, beta=beta,
                                             num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                             batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                             verbose=False, exp_id=exp_id)

            logger.info(f'{str(c_index_list)}, avg: {c_index}')
            df.loc[len(df)] = [i, num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta, c_index]
            df.to_csv(f'{logdir}/{exp_name}_results.csv', index=False)
            if (c_index > best_c_index):
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_num_layers = num_layers
                best_num_nodes = num_nodes
                best_act = act
                best_l2w = l2w
                best_dropoutp = dropoutp
                best_learning_rate = learning_rate
                best_alpha = alpha
                best_beta = beta
        else:
            logger.info('Already Selected')
            num_selected += 1
        num_layers = random.choice([1, 2, 3, 4])
        num_nodes = random.choice([4, 8, 16, 32, 64, 128])
        act = 'relu'
        l2w = random.choice([0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1])
        dropoutp = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        learning_rate = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(4,1,-1, dtype=float))]))
        alpha = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(6,-2,-1, dtype=float))]))
        beta = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(6,-2,-1, dtype=float))]+[[0]]))

    return best_num_layers, best_num_nodes, best_act, best_l2w, best_dropoutp, best_learning_rate, best_alpha, best_beta, best_i, best_c_index


def random_search_coxnam(ds, k_runs=1, df=None, no_change=10, exp_name='coxnam', logdir='logs', exp_id='exp', batch_size=2000, epochs=1000000, patience=100, early_stopping=True):
    logger = configure_logger(exp_name, logdir)
    if df is None:
        df = pd.DataFrame(
            columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'c_index'])

    selected = []
    # initial values
    num_layers = 4
    num_nodes = 16
    act = 'relu'
    l2w = 0.01
    dropoutp = 0.1
    learning_rate = 0.001

    best_c_index = 0 if df['c_index'].max() is np.nan else df['c_index'].max() #0

    best_num_layers = num_layers
    best_num_nodes = num_nodes
    best_act = act
    best_l2w = l2w
    best_dropoutp = dropoutp
    best_learning_rate = learning_rate

    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 1000)):
        i += 1
        logger.info(
            f'{i} - Testing num_layers: {num_layers}, num_nodes: {num_nodes}, act: {act}, l2w: {l2w}, dropoutp: {dropoutp}, learning_rate: {learning_rate}')

        if (num_layers, num_nodes, act, l2w, dropoutp, learning_rate) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((num_layers, num_nodes, act, l2w, dropoutp, learning_rate))
            counter += 1

            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                              batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                              verbose=False, exp_id=exp_id)

            logger.info(f'{str(c_index_list)}, avg: {c_index}')
            df.loc[len(df)] = [i, num_layers, num_nodes, act, l2w, dropoutp, learning_rate, c_index]
            df.to_csv(f'{logdir}/{exp_name}_results.csv', index=False)
            if (c_index > best_c_index):
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_num_layers = num_layers
                best_num_nodes = num_nodes
                best_act = act
                best_l2w = l2w
                best_dropoutp = dropoutp
                best_learning_rate = learning_rate
        else:
            logger.info('Already Selected')
            num_selected += 1
        num_layers = random.choice([1, 2, 3, 4])
        num_nodes = random.choice([4, 8, 16, 32, 64, 128])
        act = 'relu'
        l2w = random.choice([0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1])
        dropoutp = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        learning_rate = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(4,1,-1, dtype=float))]))

    return best_num_layers, best_num_nodes, best_act, best_l2w, best_dropoutp, best_learning_rate, best_i, best_c_index


def random_search_coxsenam(ds, k_runs=1, df=None, no_change=10, exp_name='coxsenam', logdir='logs', exp_id='exp', batch_size=2000, epochs=1000000, patience=100, early_stopping=True):
    logger = configure_logger(exp_name, logdir)
    if df is None:
        df = pd.DataFrame(
            columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'alpha', 'beta',
                     'c_index'])

    selected = []
    # initial values
    num_layers = 4
    num_nodes = 16
    act = 'relu'
    l2w = 0.01
    dropoutp = 0.1
    learning_rate = 0.001
    alpha = 0.1
    beta = 0

    best_c_index = 0 if df['c_index'].max() is np.nan else df['c_index'].max() #0

    best_num_layers = num_layers
    best_num_nodes = num_nodes
    best_act = act
    best_l2w = l2w
    best_dropoutp = dropoutp
    best_learning_rate = learning_rate
    best_alpha = alpha
    best_beta = beta

    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 1000)):
        i += 1
        logger.info(
            f'{i} - Testing num_layers: {num_layers}, num_nodes: {num_nodes}, act: {act}, l2w: {l2w}, dropoutp: {dropoutp}, learning_rate: {learning_rate}, alpha: {alpha}, beta: {beta}')

        if (num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta))
            counter += 1

            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=alpha, beta=beta,
                                                num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                                batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                                verbose=False, exp_id=exp_id)

            logger.info(f'{str(c_index_list)}, avg: {c_index}')
            df.loc[len(df)] = [i, num_layers, num_nodes, act, l2w, dropoutp, learning_rate, alpha, beta, c_index]
            df.to_csv(f'{logdir}/{exp_name}_results.csv', index=False)
            if (c_index > best_c_index):
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_num_layers = num_layers
                best_num_nodes = num_nodes
                best_act = act
                best_l2w = l2w
                best_dropoutp = dropoutp
                best_learning_rate = learning_rate
                best_alpha = alpha
                best_beta = beta
        else:
            logger.info('Already Selected')
            num_selected += 1
        num_layers = random.choice([1, 2, 3, 4])
        num_nodes = random.choice([4, 8, 16, 32, 64, 128])
        act = 'relu'
        l2w = random.choice([0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 1])
        dropoutp = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        learning_rate = random.choice(np.concatenate([x * np.arange(1, 11) for x in 10 ** (-np.arange(4, 1, -1, dtype=float))]))
        alpha = random.choice(np.concatenate([x * np.arange(1, 11) for x in 10 ** (-np.arange(6, -2, -1, dtype=float))]))
        beta = random.choice(np.concatenate([x * np.arange(1, 11) for x in 10 ** (-np.arange(6, -2, -1, dtype=float))]+[[0]]))

    return best_num_layers, best_num_nodes, best_act, best_l2w, best_dropoutp, best_learning_rate, best_alpha, best_beta, best_i, best_c_index


def random_search_cph(ds, k_runs=1, df=None, no_change=10, exp_name='surv', logdir='logs', exp_id='exp', batch_size=2000, epochs=1000000, patience=100, early_stopping=True):
    logger = configure_logger(exp_name, logdir)
    if df is None:
        df = pd.DataFrame(columns=['i', 'l2w', 'learning_rate', 'c_index'])

    selected = []
    # initial values
    l2w=0.01
    learning_rate=0.001

    best_c_index = 0 if df['c_index'].max() is np.nan else df['c_index'].max() #0

    best_l2w = l2w
    best_learning_rate = learning_rate

    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 1000)):
        i += 1
        logger.info(
            f'{i} - Testing l2w: {l2w}, learning_rate: {learning_rate}')

        if (l2w, learning_rate) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((l2w, learning_rate))
            counter += 1

            c_index, c_index_list = cph_cv(ds, k_runs=k_runs,
                                           l2w=l2w, learning_rate=learning_rate,
                                           batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                           verbose=False, exp_id=exp_id)

            logger.info(f'{str(c_index_list)}, avg: {c_index}')
            df.loc[len(df)] = [i, l2w, learning_rate, c_index]
            df.to_csv(f'{logdir}/{exp_name}_results.csv', index=False)
            if (c_index > best_c_index):
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_l2w = l2w
                best_learning_rate = learning_rate
        else:
            logger.info('Already Selected')
            num_selected += 1

        l2w = 0
        learning_rate = random.choice(np.concatenate([x*np.arange(1,11) for x in 10**(-np.arange(4,1,-1, dtype=float))]))

    return best_l2w, best_learning_rate, best_i, best_c_index
# endregion hp search functions


if __name__ == '__main__':
    exp_id = 'tunning'
    no_change = 100
    ds_size = 50000
    k_runs = 1 # number of runs per fold

    batch_size = 2000
    epochs = 1000000
    patience = 100
    early_stopping = True

    models_to_run = ['CoxSE', 'CoxNAM', 'DeepSurv', 'CoxSENAM', 'CPH']

    ds_to_run = [SimLinearPH(test_fract=0.3, n=ds_size, m=3),
                 SimNonLinearPH3(intaract_w=0, test_fract=0.3, n=ds_size, m=3),
                 SimNonLinearPH3(intaract_w=3, test_fract=0.3, n=ds_size, m=3),
                 Flchain('Dataset/flchain.csv', test_fract=0.3, verbose=True),
                 SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
                 ]

    for ds in ds_to_run:
        if 'CoxSE' in models_to_run:
            exp_name = f'{ds.get_dataset_name()}_coxse'
            df = pd.DataFrame(columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'alpha', 'beta', 'c_index'])
            try:
                df = pd.read_csv(f'logs/{exp_name}_results.csv')
                print('file read')
            except:
                print('file not found')
            random_search_coxse(ds=ds, k_runs=k_runs, df=df, no_change=no_change, exp_name=exp_name, logdir='logs',
                                exp_id=exp_id, batch_size=batch_size, epochs=epochs, patience=patience,
                                early_stopping=early_stopping)

        if 'CoxSENAM' in models_to_run:
            exp_name = f'{ds.get_dataset_name()}_coxsenam'
            df = pd.DataFrame(columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'alpha', 'beta', 'c_index'])
            try:
                df = pd.read_csv(f'logs/{exp_name}_results.csv')
                print('file read')
            except:
                print('file not found')
            random_search_coxsenam(ds=ds, k_runs=k_runs, df=df, no_change=no_change, exp_name=exp_name, logdir='logs',
                                   exp_id=exp_id, batch_size=batch_size, epochs=epochs, patience=patience,
                                   early_stopping=early_stopping)

        if 'CoxNAM' in models_to_run:
            exp_name = f'{ds.get_dataset_name()}_coxnam'
            df = pd.DataFrame(columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'c_index'])
            try:
                df = pd.read_csv(f'logs/{exp_name}_results.csv')
                print('file read')
            except:
                print('file not found')
            random_search_coxnam(ds=ds, k_runs=k_runs, df=df, no_change=no_change, exp_name=exp_name, logdir='logs',
                                 exp_id=exp_id, batch_size=batch_size, epochs=epochs, patience=patience,
                                 early_stopping=early_stopping)

        if 'DeepSurv' in models_to_run:
            exp_name = f'{ds.get_dataset_name()}_deepsurv'
            df = pd.DataFrame(columns=['i', 'num_layers', 'num_nodes', 'act', 'l2w', 'dropoutp', 'learning_rate', 'c_index'])
            try:
                df = pd.read_csv(f'logs/{exp_name}_results.csv')
                print('file read')
            except:
                print('file not found')
            random_search_deepsurv(ds=ds, k_runs=k_runs, df=df, no_change=no_change, exp_name=exp_name, logdir='logs',
                                   exp_id=exp_id, batch_size=batch_size, epochs=epochs, patience=patience,
                                   early_stopping=early_stopping)

        if 'CPH' in models_to_run:
            exp_name = f'{ds.get_dataset_name()}_cph'
            df = pd.DataFrame(columns=['i', 'l2w', 'learning_rate', 'c_index'])
            try:
                df = pd.read_csv(f'logs/{exp_name}_results.csv')
                print('file read')
            except:
                print('file not found')
            random_search_cph(ds=ds, k_runs=k_runs, df=df, no_change=no_change, exp_name=exp_name, logdir='logs',
                              exp_id=exp_id, batch_size=batch_size, epochs=epochs, patience=patience,
                              early_stopping=early_stopping)
