import numpy as np
import pandas as pd
from Models.model import CoxSE, CoxSENAM, CoxNAM, DeepSurv, CPH

import tensorflow as tf

from Utils.metrics import c_index_decomposition

from Models.model import SurvivalModelBase
from lifelines.utils.concordance import concordance_index


import shap
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os


def get_shap(mdl, x, x_background):
    new_input = Input(batch_shape=(x.shape[0], x.shape[1]))
    new_output = mdl(new_input)
    new_model = Model(new_input, new_output)

    # shap for deepSurv
    explainer = shap.DeepExplainer(new_model, x_background)
    sh = explainer.shap_values(x)
    return sh


def get_shap_x(mdl, x, x_background):
    new_input = Input(shape=(x.shape[1],))
    new_output, *_ = mdl(new_input)
    new_output = tf.keras.layers.Flatten()(new_output)
    new_model = Model(new_input, new_output)

    # shap for CoxSE
    explainer = shap.DeepExplainer(new_model, x_background)
    sh = explainer.shap_values(x)
    return sh


# region fit functions

def coxse(ds, k=0, alpha=0, beta=0, test_id=0, val_id=1,
          num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
          batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
          verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    sub_exp_name = f'CoxSE'

    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    alpha = alpha
    beta = beta

    input_shape = ds.input_shape

    verbose = verbose

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    callbacks = []

    if early_stopping:
        callbacks.append(es)

    print('Training..', alpha)

    mdl_dsx = CoxSE(input_shape=input_shape, alpha=alpha, beta=beta, gamma=0, num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp)
    mdl_dsx.special_compile(custom_optimizer=optimizer, custom_metric_func=SurvivalModelBase.cindex,
                            custom_metric_name='CI')

    mdl_dsx_history = None
    if do_fit:
        mdl_dsx_history = mdl_dsx.fit(x_train, ye_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(x_val, ye_val), callbacks=callbacks, verbose=verbose)

    y_pred_test, w_pred_test = mdl_dsx.predict(x_test)
    y_pred_train, w_pred_train = mdl_dsx.predict(x_train)
    if final_test:
        p = f'results/exp_{exp_id}'
        if not os.path.exists(p):
            os.makedirs(p)

        # ==============================================================================================================
        background_size = 1000
        shap_values_test_ds = get_shap_x(mdl=mdl_dsx, x=x_test, x_background=x_train[0:background_size])
        shap_values_train_ds = get_shap_x(mdl=mdl_dsx, x=x_train, x_background=x_train[0:background_size])

        sh_pred_test = shap_values_test_ds[0]  # .values
        sh_pred_train = shap_values_train_ds[0]  # .values
        df_test_sh = pd.DataFrame(sh_pred_test, columns=ds.features_names)
        df_train_sh = pd.DataFrame(sh_pred_train, columns=ds.features_names)
        df_test_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_sh_{val_id}_{k}.csv', index=False)
        df_train_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_sh_{val_id}_{k}.csv', index=False)
        # ==============================================================================================================
        df_test = pd.DataFrame(w_pred_test, columns=ds.features_names)
        df_train = pd.DataFrame(w_pred_train, columns=ds.features_names)
        df_test.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_{val_id}_{k}.csv', index=False)
        df_train.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_{val_id}_{k}.csv', index=False)
    ci = concordance_index(y_test, -y_pred_test, e_test)
    return ci, mdl_dsx, mdl_dsx_history


def coxnam(ds, k=0, test_id=0, val_id=1,
           num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
           batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
           verbose=False, final_test=False, do_fit=True, exp_id='exp'):

    sub_exp_name = f'CoxNAM'
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    input_shape = ds.input_shape

    verbose = verbose

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    callbacks = []

    if early_stopping:
        callbacks.append(es)

    mdl_coxnam = CoxNAM(input_shape=input_shape, num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp)
    mdl_coxnam.special_compile(custom_optimizer=optimizer, custom_metric_func=SurvivalModelBase.cindex,
                               custom_metric_name='CI')
    mdl_coxnam_history = None
    if do_fit:
        mdl_coxnam_history = mdl_coxnam.fit(x_train, ye_train, epochs=epochs, batch_size=batch_size,
                                            validation_data=(x_val, ye_val), callbacks=callbacks, verbose=verbose)

    y_pred_test, w_pred_test = mdl_coxnam.predict(x_test)
    y_pred_train, w_pred_train = mdl_coxnam.predict(x_train)
    if final_test:
        p = f'results/exp_{exp_id}'
        if not os.path.exists(p):
            os.makedirs(p)

        # ==============================================================================================================
        background_size = 1000
        shap_values_test_ds = get_shap_x(mdl=mdl_coxnam, x=x_test, x_background=x_train[0:background_size])
        shap_values_train_ds = get_shap_x(mdl=mdl_coxnam, x=x_train, x_background=x_train[0:background_size])

        sh_pred_test = shap_values_test_ds[0]
        sh_pred_train = shap_values_train_ds[0]
        df_test_sh = pd.DataFrame(sh_pred_test, columns=ds.features_names)
        df_train_sh = pd.DataFrame(sh_pred_train, columns=ds.features_names)
        df_test_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_sh_{val_id}_{k}.csv', index=False)
        df_train_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_sh_{val_id}_{k}.csv', index=False)
        # ==============================================================================================================
        df_test = pd.DataFrame(w_pred_test, columns=ds.features_names)
        df_train = pd.DataFrame(w_pred_train, columns=ds.features_names)
        df_test.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_{val_id}_{k}.csv', index=False)
        df_train.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_{val_id}_{k}.csv', index=False)
    ci = concordance_index(y_test, -y_pred_test, e_test)
    return ci, mdl_coxnam, mdl_coxnam_history


def cph(ds, k=0, test_id=0, val_id=1, l2w=0.01, learning_rate=0.001,
        batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
        verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    sub_exp_name = f'cph'
    alpha = 'cph'
    beta = 'cph'
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    input_shape = ds.input_shape

    verbose = verbose

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    callbacks = []

    if early_stopping:
        callbacks.append(es)

    print('Training..', alpha)

    mdl_cph = CPH(input_shape=input_shape, l2w=l2w)
    mdl_cph.special_compile(custom_optimizer=optimizer, custom_metric_func=SurvivalModelBase.cindex,
                            custom_metric_name='CI')
    mdl_cph_history = None
    if do_fit:
        mdl_cph_history = mdl_cph.fit(x_train, ye_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(x_val, ye_val), callbacks=callbacks, verbose=verbose)

    y_pred_test = mdl_cph.predict(x_test)
    w_pred_test = mdl_cph.feature_importance_[np.newaxis]

    if final_test:
        p = f'results/exp_{exp_id}'
        if not os.path.exists(p):
            os.makedirs(p)

        # ==============================================================================================================
        background_size = 1000
        shap_values_test_ds = get_shap(mdl=mdl_cph, x=x_test, x_background=x_train[0:background_size])
        shap_values_train_ds = get_shap(mdl=mdl_cph, x=x_train, x_background=x_train[0:background_size])

        sh_pred_test = shap_values_test_ds[0]
        sh_pred_train = shap_values_train_ds[0]
        df_test_sh = pd.DataFrame(sh_pred_test, columns=ds.features_names)
        df_train_sh = pd.DataFrame(sh_pred_train, columns=ds.features_names)
        df_test_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_sh_{val_id}_{k}.csv', index=False)
        df_train_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_sh_{val_id}_{k}.csv', index=False)
        # ==============================================================================================================
        df = pd.DataFrame(w_pred_test, columns=ds.features_names)
        df.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_{val_id}_{k}.csv', index=False)
        df.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_{val_id}_{k}.csv', index=False)
    ci = concordance_index(y_test, -y_pred_test, e_test)
    return ci, mdl_cph, mdl_cph_history


def deepsurv(ds, k=0, test_id=0, val_id=1,
             num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
             batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
             verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    sub_exp_name = f'DeepSurv'
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    input_shape = ds.input_shape

    verbose = verbose

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    callbacks = []

    if early_stopping:
        callbacks.append(es)

    mdl_ds = DeepSurv(input_shape=input_shape, num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp)
    mdl_ds.special_compile(custom_optimizer=optimizer, custom_metric_func=SurvivalModelBase.cindex,
                           custom_metric_name='CI')
    mdl_ds_history = None
    if do_fit:
        mdl_ds_history = mdl_ds.fit(x_train, ye_train, epochs=epochs, batch_size=batch_size,
                                    validation_data=(x_val, ye_val), callbacks=callbacks, verbose=verbose)

    y_pred_test = mdl_ds.predict(x_test)

    if final_test:
        p = f'results/exp_{exp_id}'
        if not os.path.exists(p):
            os.makedirs(p)
        # shap for deepSurv
        background_size = 1000

        shap_values_test_ds = get_shap(mdl=mdl_ds, x=x_test, x_background=x_train[0:background_size])
        shap_values_train_ds = get_shap(mdl=mdl_ds, x=x_train, x_background=x_train[0:background_size])

        w_pred_test = shap_values_test_ds[0]
        w_pred_train = shap_values_train_ds[0]
        df_test = pd.DataFrame(w_pred_test, columns=ds.features_names)
        df_train = pd.DataFrame(w_pred_train, columns=ds.features_names)
        df_test.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_{val_id}_{k}.csv', index=False)
        df_train.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_{val_id}_{k}.csv', index=False)

    ci = concordance_index(y_test, -y_pred_test, e_test)
    return ci, mdl_ds, mdl_ds_history


def coxsenam(ds, k=0, alpha=0, beta=0, test_id=0, val_id=1,
             num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
             batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
             verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    sub_exp_name = f'CoxSENAM'

    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    alpha = alpha
    beta = beta

    input_shape = ds.input_shape

    verbose = verbose

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    callbacks = []

    if early_stopping:
        callbacks.append(es)

    print('Training..', alpha)

    mdl_senam = CoxSENAM(input_shape=input_shape, alpha=alpha, beta=beta, num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp)
    mdl_senam.special_compile(custom_optimizer=optimizer, custom_metric_func=SurvivalModelBase.cindex,
                              custom_metric_name='CI')
    mdl_senam_history = None
    if do_fit:
        mdl_senam_history = mdl_senam.fit(x_train, ye_train, epochs=epochs, batch_size=batch_size,
                                          validation_data=(x_val, ye_val), callbacks=callbacks, verbose=verbose)

    y_pred_test, w_pred_test = mdl_senam.predict(x_test)
    y_pred_train, w_pred_train = mdl_senam.predict(x_train)
    if final_test:
        p = f'results/exp_{exp_id}'
        if not os.path.exists(p):
            os.makedirs(p)

        # ==============================================================================================================
        background_size = 1000
        shap_values_test_ds = get_shap_x(mdl=mdl_senam, x=x_test, x_background=x_train[0:background_size])
        shap_values_train_ds = get_shap_x(mdl=mdl_senam, x=x_train, x_background=x_train[0:background_size])

        sh_pred_test = shap_values_test_ds[0]
        sh_pred_train = shap_values_train_ds[0]
        df_test_sh = pd.DataFrame(sh_pred_test, columns=ds.features_names)
        df_train_sh = pd.DataFrame(sh_pred_train, columns=ds.features_names)
        df_test_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_sh_{val_id}_{k}.csv', index=False)
        df_train_sh.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_sh_{val_id}_{k}.csv', index=False)
        # ==============================================================================================================
        df_test = pd.DataFrame(w_pred_test, columns=ds.features_names)
        df_train = pd.DataFrame(w_pred_train, columns=ds.features_names)
        df_test.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_test_predictions_{val_id}_{k}.csv', index=False)
        df_train.to_csv(f'results/exp_{exp_id}/{sub_exp_name}_{ds.get_dataset_name()}_train_predictions_{val_id}_{k}.csv', index=False)

    ci = concordance_index(y_test, -y_pred_test, e_test)
    return ci, mdl_senam, mdl_senam_history
# endregion fit functions


# region cv functions
def coxse_cv(ds, k_runs=1, alpha=0, beta=0,
             num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
             batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
             verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    cis = []

    if final_test:
        ids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    else:
        ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for i, j in ids:
        ci_k_runs = []
        for k in range(k_runs):
            ci_k, mdl_k, history_k = coxse(ds, k=k, alpha=alpha, beta=beta, test_id=i, val_id=j,
                                           num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                           batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                           verbose=verbose, final_test=final_test, do_fit=do_fit, exp_id=exp_id)
            ci_k_runs.append(ci_k)
            print(f'Fold {j}-Run {k}: {ci_k}')
        cis.append(np.mean(ci_k_runs))
    print(np.mean(cis), np.std(cis))
    return np.mean(cis), cis


def coxnam_cv(ds, k_runs=1,
              num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
              batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
              verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    cis = []

    if final_test:
        ids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    else:
        ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for i, j in ids:
        ci_k_runs = []
        for k in range(k_runs):
            ci_k, mdl_k, history_k = coxnam(ds, k=k, test_id=i, val_id=j,
                                            num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                            batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                            verbose=verbose, final_test=final_test, do_fit=do_fit, exp_id=exp_id)
            ci_k_runs.append(ci_k)
            print(f'Fold {j}-Run {k}: {ci_k}')
        cis.append(np.mean(ci_k_runs))
    print(np.mean(cis), np.std(cis))
    return np.mean(cis), cis


def deepsurv_cv(ds, k_runs=1,
                num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
                batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
                verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    cis = []

    if final_test:
        ids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    else:
        ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for i, j in ids:
        ci_k_runs = []
        for k in range(k_runs):
            ci_k, mdl_k, history_k = deepsurv(ds, k=k, test_id=i, val_id=j,
                                              num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                              batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                              verbose=verbose, final_test=final_test, do_fit=do_fit, exp_id=exp_id)
            ci_k_runs.append(ci_k)
            print(f'Fold {j}-Run {k}: {ci_k}')
        cis.append(np.mean(ci_k_runs))
    print(np.mean(cis), np.std(cis))
    return np.mean(cis), cis


def coxsenam_cv(ds, k_runs=1, alpha=0, beta=0,
                num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, learning_rate=0.001,
                batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
                verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    cis = []


    if final_test:
        ids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    else:
        ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for i, j in ids:
        ci_k_runs = []
        for k in range(k_runs):
            ci_k, mdl_k, history_k = coxsenam(ds, k=k, alpha=alpha, beta=beta, test_id=i, val_id=j,
                                              num_layers=num_layers, num_nodes=num_nodes, act=act, l2w=l2w, dropoutp=dropoutp, learning_rate=learning_rate,
                                              batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                              verbose=verbose, final_test=final_test, do_fit=do_fit, exp_id=exp_id)
            ci_k_runs.append(ci_k)
            print(f'Fold {j}-Run {k}: {ci_k}')
        cis.append(np.mean(ci_k_runs))
    print(np.mean(cis), np.std(cis))
    return np.mean(cis), cis


def cph_cv(ds, k_runs=1, l2w=0.01, learning_rate=0.001,
           batch_size=2000, epochs=1000000, patience=100, early_stopping=True,
           verbose=False, final_test=False, do_fit=True, exp_id='exp'):
    cis = []

    if final_test:
        ids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    else:
        ids = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for i, j in ids:
        ci_k_runs = []
        for k in range(k_runs):
            ci_k, mdl_k, history_k = cph(ds, k=k, test_id=i, val_id=j, l2w=l2w, learning_rate=learning_rate,
                                         batch_size=batch_size, epochs=epochs, patience=patience, early_stopping=early_stopping,
                                         verbose=verbose, final_test=final_test, do_fit=do_fit, exp_id=exp_id)
            ci_k_runs.append(ci_k)
            print(f'Fold {j}-Run {k}: {ci_k}')
        cis.append(np.mean(ci_k_runs))
    print(np.mean(cis), np.std(cis))
    return np.mean(cis), cis
# endregion cv functions


