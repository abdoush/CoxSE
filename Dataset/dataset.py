import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from itertools import permutations
from sklearn.model_selection import train_test_split
from .relative_risk import SimStudyLinearPH, SimStudyNonLinearPH


class Dataset:
    def __init__(self, dataset_file_path=None, number_of_splits=5, test_fract=0,
                 p='full', events_only=True,  action='censor', drop_feature=None, normalize_target=True,
                 random_seed=20, verbose=False):
        self.dataset_file_path = dataset_file_path
        self.number_of_splits = number_of_splits
        self.p = p # if events_only=True then p is the events percentage, else it is the percentage to drop from the whole data
        self.normalize_target = normalize_target
        self.events_only = events_only
        self.action = action  # action='censor' or 'drop'
        self.drop_feature = drop_feature
        self.df = self._load_data()
        self.rest_df, self.test_df = self._get_test_split(fract=test_fract, seed=random_seed)
        self.n_splits = self._get_n_splits(seed=random_seed)
        self.input_shape = self.rest_df.shape[1] - 2
        self.features_names = list(self.df.columns.drop(['T', 'E']))
        if verbose:
            self.print_dataset_summery()

    def get_dataset_name(self):
        pass

    def _preprocess_x(self, x_df):
        pass

    def _preprocess_y(self, y_df, normalizing_val=None):
        pass

    def _preprocess_e(self, e_df):
        pass

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None):
        pass

    def _load_data(self):
        pass

    def get_x_dim(self):
        return self.df.shape[1]-2

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        scaler = MinMaxScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

    def print_dataset_summery(self):
        s = 'Dataset Description =======================\n'
        s += 'Dataset Name: {}\n'.format(self.get_dataset_name())
        s += 'Dataset Shape: {}\n'.format(self.df.shape)
        s += 'Events: %.2f %%\n' % (self.df['E'].sum()*100 / len(self.df))
        s += 'NaN Values: %.2f %%\n' % (self.df.isnull().sum().sum()*100 / self.df.size)
        s += 'Size and Events % in splits: '
        for split in self.n_splits:
            s += '({}, {:.2f}%), '.format((split.shape[0]), (split["E"].mean()*100))
        s += '\n'
        if self.test_df is not None:
            s += '-------------------------------------------\n'
            s += 'Hold-out Testset % of Data: {:.2f}%\n'.format((self.test_df.shape[0] * 100 / self.df.shape[0]))
            s += 'Hold-out Testset Size and Events %: ({:}, {:.2f}%) \n'.format(self.test_df.shape[0], (self.test_df["E"].mean()*100))
        s += '===========================================\n'
        print(s)
        return s

    @staticmethod
    def max_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = ((df_transformed[col]) / df_transformed[col].max()) ** powr
        return df_transformed

    @staticmethod
    def log_transform(df, cols):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = np.abs(np.log(df_transformed[col] + 1e-8))
        return df_transformed

    @staticmethod
    def power_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = df_transformed[col] ** powr
        return df_transformed

    def _get_test_split(self, fract=0.4, seed=20):
        if fract == 0:
            return self.df, None
        rest_df, test_df = train_test_split(self.df, test_size=fract, random_state=seed, shuffle=True, stratify=self.df['E'])
        return rest_df, test_df

    def _get_n_splits(self, seed=20):
        k = self.number_of_splits
        train_df = self.rest_df
        df_splits = []
        for i in range(k, 1, -1):
            train_df, test_df = train_test_split(train_df, test_size=(1 / i), random_state=seed, shuffle=True,
                                                 stratify=train_df['E'])
            df_splits.append(test_df)
            if i == 2:
                df_splits.append(train_df)
        return df_splits

    def get_train_val_test_final_eval(self, val_id):
        if self.test_df is None:
            print('No hold-out test set found')
            return
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = self.test_df
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_from_splits(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_from_splits(self, val_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)

        self._fill_missing_values(x_train_df, x_val_df)

        x_train, x_val = self._preprocess_x(x_train_df), self._preprocess_x(x_val_df)

        x_train, x_val = self._scale_x(x_train, x_val)

        y_train, y_val = self._preprocess_y(y_train_df), self._preprocess_y(y_val_df)

        e_train, e_val = self._preprocess_e(e_train_df), self._preprocess_e(e_val_df)

        ye_train, ye_val = np.array(list(zip(y_train, e_train))), np.array(list(zip(y_val, e_val)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val)

    def _formatted_data(self, x, t, e):
        survival_data = {'x': x, 't': t, 'e': e}
        return survival_data

    @staticmethod
    def _split_columns(df):
        y_df = df['T']
        e_df = df['E']
        x_df = df.drop(['T', 'E'], axis=1)
        return x_df, y_df, e_df

    def test_dataset(self):
        combs = list(permutations(range(self.number_of_splits), 2))
        for i, j in combs:
            (x_train, ye_train, y_train, e_train,
             x_val, ye_val, y_val, e_val,
             x_test, ye_test, y_test, e_test) = self.get_train_val_test_from_splits(i, j)
            assert np.isnan(x_train).sum() == 0
            assert np.isnan(x_val).sum() == 0
            assert np.isnan(x_test).sum() == 0


# region Real Datasets

class Flchain(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df.drop('chapter', axis=1, inplace=True)
        df['sample.yr'] = df['sample.yr'].astype('category')
        df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        self.covariates = [x for x in df.columns if x not in ['T', 'E']]
        ohdf = pd.get_dummies(df)
        return ohdf

    def get_dataset_name(self):
        return 'flchain'

    # def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
    #     m = x_train_df['creatinine'].median()
    #     x_train_df['creatinine'].fillna(m, inplace=True)
    #     x_val_df['creatinine'].fillna(m, inplace=True)
    #     if x_test_df is not None:
    #         x_test_df['creatinine'].fillna(m, inplace=True)
    #     if x_tune_df is not None:
    #         x_tune_df['creatinine'].fillna(m, inplace=True)

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None):
        m = x_train_df['creatinine'].median()
        x_train_df['creatinine'].fillna(m, inplace=True)
        x_val_df['creatinine'].fillna(m, inplace=True)
        if x_test_df is not None:
            x_test_df['creatinine'].fillna(m, inplace=True)

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if self.normalize_target:
            if normalizing_val is None:
                normalizing_val = y_df.max()
            return ((y_df / normalizing_val).values ** 0.5).astype('float32')
        else:
            return ((y_df).values).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.values.astype('float32')

    # def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
    #     #scaler = StandardScaler().fit(x_train_df)
    #     scaler = MinMaxScaler().fit(x_train_df)
    #     x_train = scaler.transform(x_train_df)
    #     x_val = scaler.transform(x_val_df)
    #     if (x_tune_df is not None) & (x_test_df is not None):
    #         x_test = scaler.transform(x_test_df)
    #         x_tune = scaler.transform(x_tune_df)
    #         return x_train, x_val, x_test, x_tune
    #     elif x_test_df is not None:
    #         x_test = scaler.transform(x_test_df)
    #         return x_train, x_val, x_test
    #     else:
    #         return x_train, x_val

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        # scaler = StandardScaler().fit(x_train_df)
        scaler = MinMaxScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class SEERBreastCancer(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df.drop(columns=['patient_id'], inplace=True)
        df.rename(columns={'survival_months': 'T', 'event': 'E'}, inplace=True)
        return df

    def get_dataset_name(self):
        return 'SEERBreastCancer'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 1.0).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        scaler = MinMaxScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

# endregion Real Datasets


# region Simulated Datasets

class SimStudyNonLinearPH3(SimStudyNonLinearPH):
    def g(self, covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        print(f'3 * x0 ** 2 + 2 * x1 ** 2 + x2 ** 2 + {self.intaract_w} * x0 * x1')
        nonlinear = 3 * x0 ** 2 + 2 * x1 ** 2 + x2 ** 2 + self.intaract_w * x0 * x1
        return nonlinear


class SimLinearPH(Dataset):
    def __init__(self, n=20000, m=3, x_seed=12345, t_seed=12345, *args, **kwargs):
        self.n = n
        self.m = m
        self.x_seed = x_seed
        self.t_seed = t_seed
        super().__init__(*args, **kwargs)

    def _load_data(self):
        sim_mdl = SimStudyLinearPH(x_seed=self.x_seed, t_seed=self.t_seed)
        data = sim_mdl.simulate(n=self.n, m=self.m)

        x, t, e = data['covs'], data['durations'], data['events']
        df = pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])
        df['T'] = t
        df['E'] = e
        return df

    def get_dataset_name(self):
        return 'SimStudyLinearPH'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        scaler = MinMaxScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class SimNonLinearPH3(Dataset):
    def __init__(self, intaract_w=0, n=20000, m=3, x_seed=12345, t_seed=12345, *args, **kwargs):
        self.intaract_w = intaract_w
        self.n = n
        self.m = m
        self.x_seed = x_seed
        self.t_seed = t_seed
        super().__init__(*args, **kwargs)

    def _load_data(self):
        sim_mdl = SimStudyNonLinearPH3(self.intaract_w, x_seed=self.x_seed, t_seed=self.t_seed)
        data = sim_mdl.simulate(n=self.n, m=self.m)

        x, t, e = data['covs'], data['durations'], data['events']
        df = pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])
        df['T'] = t
        df['E'] = e
        return df

    def get_dataset_name(self):
        return f'SimNonLinearPH3_{int(self.intaract_w)}'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        scaler = MinMaxScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

# endregion Simulated Datasets
