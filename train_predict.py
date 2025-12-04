import pandas as pd
from Dataset.dataset import Flchain, SimLinearPH, SimNonLinearPH3, SEERBreastCancer
from Models.train_helper import coxse_cv, deepsurv_cv, coxnam_cv, coxsenam_cv, cph_cv
import os


def logvalues(logdf, dataset_name, model_name, values):
    for i, val in enumerate(values):
        logdf.loc[len(logdf)] = [dataset_name, model_name, i, val]


if __name__ == '__main__':
    nfolds = 5
    no_change = 10
    k_runs = 1
    ds_size = 50000
    epochs = 1000000
    models_to_run = ['CoxSE', 'CoxNAM', 'DeepSurv', 'CoxSENAM', 'CPH']
    ds_to_run = ['SimLinearPH3', 'SimNonLinearPH3_0', 'SimNonLinearPH3_3', 'Flchain'] # 'SEERBreastCancer'

    exp_id = 'final_results'
    log_file_path = 'results/all_exp_results.csv'
    if os.path.isfile(log_file_path):
        df = pd.read_csv(log_file_path)
    else:
        df = pd.DataFrame(columns=['Dataset', 'Model', 'Fold', 'C_Index'])

    if 'CoxSE' in models_to_run:
        model_name = 'CoxSE'
        if 'SimLinearPH3' in ds_to_run:
            ds = SimLinearPH(test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=0.5, beta=0,
                                             num_layers=4, num_nodes=128, act='relu', l2w=0, dropoutp=0.4, learning_rate=0.01,
                                             batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                             verbose=False, final_test=True, do_fit=True, exp_id=exp_id)

            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_0' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=0, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=3e-06, beta=0.0005,
                                             num_layers=1, num_nodes=4, act='relu', l2w=0.5, dropoutp=0, learning_rate=0.001,
                                             batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                             verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_3' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=3, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=9.999999999999999e-06, beta=0.0001,
                                             num_layers=1, num_nodes=32, act='relu', l2w=0.5, dropoutp=0,learning_rate=0.1,
                                             batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                             verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # --------------------------------------------------------------------------------------------------------------
        if 'Flchain' in ds_to_run:
            ds = Flchain('Dataset/flchain.csv', number_of_splits=nfolds, test_fract=0.3)
            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=4e-05, beta=0.006,
                                             num_layers=4, num_nodes=4, act='relu', l2w=0.5, dropoutp=0.3, learning_rate=0.01,
                                             batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                             verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # --------------------------------------------------------------------------------------------------------------
        if 'SEERBreastCancer' in ds_to_run:
            ds = SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
            c_index, c_index_list = coxse_cv(ds, k_runs=k_runs, alpha=1e-05, beta=3e-05,
                                             num_layers=1, num_nodes=4, act='relu', l2w=0.5, dropoutp=0,learning_rate=0.0009000000000000001,
                                             batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                             verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
    # ==================================================================================================================
    if 'CoxSENAM' in models_to_run:
        model_name = 'CoxSENAM'
        if 'SimLinearPH3' in ds_to_run:
            ds = SimLinearPH(test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=0.5, beta=0.005,
                                                num_layers=1, num_nodes=4, act='relu', l2w=0.5, dropoutp=0, learning_rate=0.1,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_0' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=0, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=9e-06, beta=9e-06,
                                                num_layers=1, num_nodes=128, act='relu', l2w=1, dropoutp=0.4, learning_rate=0.006,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_3' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=3, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=0.003, beta=4.9999999999999996e-06,
                                                num_layers=1, num_nodes=64, act='relu', l2w=0, dropoutp=0.1, learning_rate=0.1,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'Flchain' in ds_to_run:
            ds = Flchain('Dataset/flchain.csv', number_of_splits=nfolds, test_fract=0.3)
            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=6e-05, beta=5e-05,
                                                num_layers=4, num_nodes=32, act='relu', l2w=0.001, dropoutp=0.4, learning_rate=0.003,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SEERBreastCancer' in ds_to_run:
            ds = SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
            c_index, c_index_list = coxsenam_cv(ds, k_runs=k_runs, alpha=7.000000000000001e-05, beta=4.9999999999999996e-06,
                                                num_layers=4, num_nodes=16, act='relu', l2w=0.5, dropoutp=0,
                                                learning_rate=0.001,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
    # ==================================================================================================================
    if 'CoxNAM' in models_to_run:
        model_name = 'CoxNAM'
        if 'SimLinearPH3' in ds_to_run:
            ds = SimLinearPH(test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=2, num_nodes=32, act='relu', l2w=0.7, dropoutp=0, learning_rate=0.1,
                                              batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                              verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_0' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=0, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=4, num_nodes=128, act='relu', l2w=1, dropoutp=0, learning_rate=0.0009000000000000001,
                                              batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                              verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_3' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=3, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=2, num_nodes=128, act='relu', l2w=0.05, dropoutp=0, learning_rate=0.05,
                                              batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                              verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'Flchain' in ds_to_run:
            ds = Flchain('Dataset/flchain.csv', number_of_splits=nfolds, test_fract=0.3)
            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=4, num_nodes=64, act='relu', l2w=0, dropoutp=0,
                                              learning_rate=0.001,
                                              batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                              verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SEERBreastCancer' in ds_to_run:
            ds = SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
            c_index, c_index_list = coxnam_cv(ds, k_runs=k_runs,
                                              num_layers=4, num_nodes=16, act='relu', l2w=0.05, dropoutp=0,
                                              learning_rate=0.0005,
                                              batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                              verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
    # ==================================================================================================================
    if 'DeepSurv' in models_to_run:
        model_name = 'DeepSurv'

        if 'SimLinearPH3' in ds_to_run:
            ds = SimLinearPH(test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=3, num_nodes=4, act='relu', l2w=0.1, dropoutp=0, learning_rate=0.05,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_0' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=0, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=1, num_nodes=128, act='relu', l2w=0.01, dropoutp=0, learning_rate=0.001,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_3' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=3, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=1, num_nodes=128, act='relu', l2w=0.5, dropoutp=0, learning_rate=0.0001,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'Flchain' in ds_to_run:
            ds = Flchain('Dataset/flchain.csv', number_of_splits=nfolds, test_fract=0.3)
            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=1, num_nodes=16, act='relu', l2w=1, dropoutp=0.1, learning_rate=0.0005, # temp
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SEERBreastCancer' in ds_to_run:
            ds = SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
            c_index, c_index_list = deepsurv_cv(ds, k_runs=k_runs,
                                                num_layers=1, num_nodes=32, act='relu', l2w=1, dropoutp=0.2,
                                                learning_rate=0.0005,
                                                batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                                verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
    # ==================================================================================================================
    if 'CPH' in models_to_run:
        model_name = 'CPH'
        if 'SimLinearPH3' in ds_to_run:
            ds = SimLinearPH(test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = cph_cv(ds, k_runs=k_runs, l2w=0.7, learning_rate=0.1,
                                           batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                           verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_0' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=0, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = cph_cv(ds, k_runs=k_runs, l2w=0, learning_rate=0.00030000000000000003,
                                           batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                           verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'SimNonLinearPH3_3' in ds_to_run:
            ds = SimNonLinearPH3(intaract_w=3, test_fract=0.3, number_of_splits=nfolds, n=ds_size, m=3)
            c_index, c_index_list = cph_cv(ds, k_runs=k_runs, l2w=0, learning_rate=0.03,
                                           batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                           verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        # ------------------------------------------------------------------------------------------------------------------
        if 'Flchain' in ds_to_run:
            ds = Flchain('Dataset/flchain.csv', number_of_splits=nfolds, test_fract=0.3)
            c_index, c_index_list = cph_cv(ds, k_runs=k_runs, l2w=0, learning_rate=0.09,
                                           batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                           verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
        if 'SEERBreastCancer' in ds_to_run:
            ds = SEERBreastCancer(dataset_file_path='Dataset/SEER_BC_processed_240331.csv', test_fract=0.3, verbose=True)
            c_index, c_index_list = cph_cv(ds, k_runs=k_runs, l2w=0.05, learning_rate=0.01,
                                           batch_size=2000, epochs=epochs, patience=100, early_stopping=True,
                                           verbose=False, final_test=True, do_fit=True, exp_id=exp_id)
            logvalues(df, dataset_name=ds.get_dataset_name(), model_name=model_name, values=c_index_list)
            df.to_csv(log_file_path, index=False)
    # ==================================================================================================================


