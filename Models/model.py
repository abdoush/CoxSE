import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class CIndex(tf.keras.metrics.Metric):

    def __init__(self, name='cindex', **kwargs):
        super(CIndex, self).__init__(name=name, **kwargs)
        self.cindex = self.add_weight(name=name, initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = SurvivalModelBase.cindex(y_true=y_true, y_pred=y_pred)
        self.cindex.assign(values)

    def result(self):
        return self.cindex

    def reset_states(self):
        self.cindex.assign(0.0)

# region general base models


class BBModel(tf.keras.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize variables for special_compile().
        self.custom_optimizer, self.custom_loss, self.custom_metric = None, None, None
        self.custom_loss_mean, self.custom_metric_mean = None, None

        self.ds_input_shape = input_shape

        self._create_model()

    def target_loss(self):
        pass

    def get_metric(self):
        pass

    def _create_model(self):
        pass

    def special_compile(self, custom_optimizer, custom_metric_func, custom_metric_name):
        self.custom_optimizer = custom_optimizer
        self.custom_metric = custom_metric_func
        self.custom_loss_mean = tf.keras.metrics.Mean(name='loss')
        self.custom_metric_mean= tf.keras.metrics.Mean(name=custom_metric_name) #tf.keras.metrics.Mean(name='cindex')(custom_metric)
        super().compile()

    def call(self, input_tensor, training=False, mask=None):
        output = self.encoder(input_tensor)
        return output

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            losses = self.target_loss()(y, y_pred)
        loss = losses

        # The training happens here.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.custom_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metric_val = self.custom_metric(y, y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(losses)
            else:
                metric.update_state(metric_val)
            # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.target_loss()(y, y_pred)
        metric_val = self.custom_metric(y, y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(metric_val)
            # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class XModel(BBModel):
    def __init__(self, alpha, beta=0, gamma=0, *args, **kwargs):

        # Initialize variables for special_compile().
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__(*args, **kwargs)

    def special_compile(self, custom_optimizer, custom_metric_func, custom_metric_name):
        self.custom_optimizer = custom_optimizer
        self.custom_metric = custom_metric_func
        self.custom_loss_mean = tf.keras.metrics.Mean(name='loss')
        self.custom_metric_mean= tf.keras.metrics.Mean(name=custom_metric_name)
        self.y_metric = tf.keras.metrics.Mean(name='y_loss')
        self.w_metric = tf.keras.metrics.Mean(name='w_loss')
        super().compile()

    def wtx(self, w, x):
        return tf.reduce_sum(tf.multiply(w, x), 1, keepdims=True, name='dotprod_layer')

    def weight_loss2(self, w_pred):
        w_norm = tf.linalg.norm(w_pred, axis=1, ord=1)
        return tf.reduce_mean(w_norm)

    def weight_loss3(self, w_pred):
        w_norm = tf.linalg.norm(w_pred, axis=1, ord=2)
        return tf.reduce_mean(w_norm)

    @staticmethod
    def weight_loss1(w_grads, w_pred):
        return tf.reduce_sum(tf.square(w_grads - w_pred), axis=1)

    def call(self, input_tensor, training=False, mask=None):
        z = self.encoder(input_tensor)
        w = self.w_decoder(z)
        output = self.wtx(w, input_tensor)
        return output, w

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            target_pred, w_pred = self(x, training=True)
            t_loss = tf.reduce_mean(self.target_loss()(y, target_pred))
            w_grads = tape.gradient(target_pred, x)
            w_loss1 = self.alpha * tf.reduce_mean(self.weight_loss1(tf.cast(w_grads, dtype=tf.float32), w_pred))
            w_loss2 = self.beta * tf.reduce_mean(self.weight_loss2(w_pred))
            w_loss3 = self.gamma * tf.reduce_mean(self.weight_loss3(w_pred))
            w_loss = w_loss1 + w_loss2 + w_loss3
            loss = t_loss + w_loss

        # The training happens here.
        gradients = tape.gradient(loss, self.trainable_variables)
        del tape
        self.custom_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Calculate metric and add it to the mean.
        metric_val = self.custom_metric(y, target_pred)

        self.custom_loss_mean.update_state(loss)
        self.y_metric.update_state(t_loss)
        self.w_metric.update_state(w_loss)
        self.custom_metric_mean.update_state(metric_val)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            tape.watch(x)
            target_pred, w_pred = self(x, training=False)
            w_grads = tape.gradient(target_pred, x)

        w_loss = 0.0
        w_loss1 = self.alpha * tf.reduce_mean(self.weight_loss1(tf.cast(w_grads, dtype=tf.float32), w_pred))
        w_loss2 = self.beta * tf.reduce_mean(self.weight_loss2(w_pred))
        w_loss3 = self.gamma * tf.reduce_mean(self.weight_loss3(w_pred))
        w_loss = w_loss1 + w_loss2 + w_loss3

        t_loss = tf.reduce_mean(self.target_loss()(y, target_pred))
        loss = t_loss + w_loss

        metric_val = self.custom_metric(y, target_pred)

        self.custom_loss_mean.update_state(loss)
        self.y_metric.update_state(t_loss)
        self.w_metric.update_state(w_loss)
        self.custom_metric_mean.update_state(metric_val)
        return {m.name: m.result() for m in self.metrics}

    def predict_y(self, x):
        y,*_ = self.predict(x)
        return y

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.custom_loss_mean)
        metrics.append(self.y_metric)
        metrics.append(self.w_metric)
        metrics.append(self.custom_metric_mean)
        return metrics


class NAM(BBModel):
    def call(self, input_tensor, training=False, mask=None):
        ws = []
        for i in range(self.ds_input_shape):
            x = tf.expand_dims(input_tensor[:,i], axis=1)
            wi = self.encoders[i](x)
            ws.append(wi)
        w = tf.stack(ws, axis=1)
        output = tf.reduce_sum(w, axis=1)
        w = tf.squeeze(w, axis=[-1])
        return output, w

    def _create_model(self):
        self.encoders = []
        for i in range(self.ds_input_shape):
            encoder = keras.Sequential(
                [
                    layers.Dense(8, activation="relu", name='layer1', kernel_regularizer='l2', input_shape=(1,)),
                    layers.Dense(8, activation="relu", kernel_regularizer='l2', name='layer2'),
                    layers.Dense(1, kernel_regularizer='l2', name='Output', use_bias=False)
                ]
            )
            self.encoders.append(encoder)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            target_pred, w_pred = self(x, training=True)
            t_loss = self.target_loss()(y, target_pred)
            loss = t_loss

        # The training happens here.
        gradients = tape.gradient(loss, self.trainable_variables)
        del tape
        self.custom_optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Calculate metric and add it to the mean.
        metric_val = self.custom_metric(y, target_pred)
        self.custom_loss_mean.update_state(loss)
        self.custom_metric_mean.update_state(metric_val)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            target_pred, w_pred = self(x, training=False)
        t_loss = self.target_loss()(y, target_pred)
        loss = t_loss
        metric_val = self.custom_metric(y, target_pred)

        self.custom_loss_mean.update_state(loss)
        self.custom_metric_mean.update_state(metric_val)
        return {m.name: m.result() for m in self.metrics}

    def predict_y(self, x):
        y,*_ = self.predict(x)
        return y


class SENAM(XModel):
    def call(self, input_tensor, training=False, mask=None):
        ws = []
        for i in range(self.ds_input_shape):
            x = tf.expand_dims(input_tensor[:,i], axis=1)
            wi = self.encoders[i](x)
            ws.append(wi)
        w = tf.squeeze(tf.stack(ws, axis=1), axis=[-1])
        output = self.wtx(w, input_tensor)
        return output, w

    def _create_model(self):
        self.encoders = []
        for i in range(self.ds_input_shape):
            encoder = keras.Sequential(
                [
                    layers.Dense(8, activation="relu", name='layer1', kernel_regularizer='l2', input_shape=(1,)),
                    layers.Dense(8, activation="relu", kernel_regularizer='l2', name='layer2'),
                    layers.Dense(1, kernel_regularizer='l2', name='Output', use_bias=False)
                ]
            )
            self.encoders.append(encoder)

# endregion base models

# region survival base models

class SurvivalModelBase:
    @staticmethod
    def neg_partial_log_likilihood(y_true, y_pred):
        eps = 1e-7
        t = y_true[:, 0]
        ids = tf.argsort(t, direction='DESCENDING')
        e = tf.gather(y_true[:, 1], ids)
        log_h = tf.gather(y_pred, ids)
        log_h = tf.reshape(log_h, [-1])
        gamma = tf.reduce_max(log_h)
        log_cumsum_h = tf.math.log(tf.math.cumsum(tf.exp(log_h - gamma)) + eps) + gamma
        part_log_likelihood = tf.reduce_sum((log_h - log_cumsum_h) * e) / tf.reduce_sum(e)
        return -part_log_likelihood

    @staticmethod
    def cindex(y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        y_pred = -y_pred
        ydiff = y[tf.newaxis, :] - y[:, tf.newaxis]
        yij = K.cast(K.greater(ydiff, 0), K.floatx()) + K.cast(K.equal(ydiff, 0), K.floatx()) * K.cast(
            e[:, tf.newaxis] != e[tf.newaxis, :], K.floatx())  # yi > yj
        is_valid_pair = yij * e[:, tf.newaxis]

        ypdiff = tf.transpose(y_pred) - y_pred
        ypij = K.cast(K.greater(ypdiff, 0), K.floatx()) + 0.5 * K.cast(K.equal(ypdiff, 0), K.floatx())  # yi > yj
        ci = (K.sum(ypij * is_valid_pair)) / K.sum(is_valid_pair)
        return ci

    def predict_survival_function(self, x):
        raise NotImplementedError("Implementation coming soon.")


class CoxNAMBase(NAM, SurvivalModelBase):
    def target_loss(self):
        return self.neg_partial_log_likilihood

    def get_metric(self):
        return self.cindex

    def _create_model(self):
        self.encoders = []
        for i in range(self.ds_input_shape):
            encoder = keras.Sequential(
                [
                    layers.Dense(8, activation="relu", name='layer1', kernel_regularizer='l2', input_shape=(1,)),
                    layers.Dense(8, activation="relu", kernel_regularizer='l2', name='layer2'),
                    layers.Dense(1, kernel_regularizer='l2', name='Output', use_bias=True)
                ]
            )
            self.encoders.append(encoder)


class CoxSENAMBase(SENAM, SurvivalModelBase):
    def target_loss(self):
        return self.neg_partial_log_likilihood

    def get_metric(self):
        return self.cindex

    def _create_model(self):
        self.encoders = []
        for i in range(self.ds_input_shape):
            encoder = keras.Sequential(
                [
                    layers.Dense(8, activation="relu", name='layer1', kernel_regularizer='l2', input_shape=(1,)),
                    layers.Dense(8, activation="relu", kernel_regularizer='l2', name='layer2'),
                    layers.Dense(1, kernel_regularizer='l2', name='Output', use_bias=True)
                ]
            )
            self.encoders.append(encoder)


class CPHBase(BBModel, SurvivalModelBase):
    def target_loss(self):
        return self.neg_partial_log_likilihood

    def get_metric(self):
        return self.cindex

    @property
    def feature_importance_(self):
        return self.encoder.layers[0].weights[0].numpy().flatten()

    def _create_model(self):
        self.encoder = keras.Sequential(
            [
                layers.Dense(1, name='D_layer1', kernel_regularizer='l2', input_shape=(self.ds_input_shape,), use_bias=False)
            ]
        )


class DeepSurvBase(BBModel, SurvivalModelBase):
    def target_loss(self):
        return self.neg_partial_log_likilihood

    def get_metric(self):
        return self.cindex

    def _create_model(self):
        x = keras.Input(shape=(self.ds_input_shape,))

        self.encoder = keras.Sequential(
            [
                layers.Dense(16, activation="relu", kernel_regularizer='l2', name='layer1',input_shape=(self.ds_input_shape,)),
                layers.Dropout(0.1),

                layers.Dense(16, activation="relu", kernel_regularizer='l2', name='layer2'),
                layers.Dropout(0.1),

                layers.Dense(1, kernel_regularizer='l2', name='Output')
            ]
        )


class CoxSEBase(XModel, SurvivalModelBase):
    def target_loss(self):
        return self.neg_partial_log_likilihood

    def get_metric(self):
        return self.cindex

    def _create_model(self):

        x = keras.Input(shape=(self.ds_input_shape,))

        self.encoder = keras.Sequential(
            [
                layers.Dense(16, activation="relu", kernel_regularizer='l2', name='layer1', input_shape=(self.ds_input_shape,)),
                layers.Dropout(0.5),

                layers.Dense(16, activation="relu", kernel_regularizer='l2', name='latent_layer'),
                layers.Dropout(0.5)
            ]
        )

        self.w_decoder = keras.Sequential(
            [
                layers.Dense(self.ds_input_shape, kernel_regularizer='l2', name='w_layer')
            ]
        )

# endregion survival base models

# region derived models with hyperparameters

class CoxSE(CoxSEBase):
    def __init__(self, num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, *args, **kwargs):

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act = act
        self.dropoutp = dropoutp
        self.l2w = l2w
        super().__init__(*args, **kwargs)

    def _create_model(self):
        tf.random.set_seed(42)
        net_layers = [keras.Input(shape=(self.ds_input_shape,))]
        for i in range(self.num_layers):
            net_layers.append(
                layers.Dense(self.num_nodes, activation=self.act, kernel_regularizer=regularizers.l2(self.l2w),
                             name=f'layer{i}'))
            net_layers.append(layers.Dropout(self.dropoutp))

        self.encoder = keras.Sequential(net_layers)

        self.w_decoder = keras.Sequential(
            [
                layers.Dense(self.ds_input_shape, name='w_layer', kernel_regularizer=regularizers.l2(self.l2w))
            ]
        )


class DeepSurv(DeepSurvBase):
    def __init__(self, num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, *args, **kwargs):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act = act
        self.dropoutp = dropoutp
        self.l2w = l2w

        super().__init__(*args, **kwargs)

    def _create_model(self):
        tf.random.set_seed(42)
        net_layers = [keras.Input(shape=(self.ds_input_shape,))]
        for i in range(self.num_layers):
            net_layers.append(
                layers.Dense(self.num_nodes, activation=self.act, kernel_regularizer=regularizers.l2(self.l2w),
                             name=f'layer{i}'))
            net_layers.append(layers.Dropout(self.dropoutp))

        net_layers.append(layers.Dense(1, name='Output'))

        self.encoder = keras.Sequential(net_layers)


class CoxNAM(CoxNAMBase):
    def __init__(self, num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, *args, **kwargs):

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act = act
        self.dropoutp = dropoutp
        self.l2w = l2w

        super().__init__(*args, **kwargs)

    def _create_model(self):
        tf.random.set_seed(42)
        self.encoders = []
        for i in range(self.ds_input_shape):
            net_layers = [keras.Input(shape=(1,))]
            for i in range(self.num_layers):
                net_layers.append(
                    layers.Dense(self.num_nodes, activation=self.act, kernel_regularizer=regularizers.l2(self.l2w),
                                 name=f'layer{i}'))
                net_layers.append(layers.Dropout(self.dropoutp))
            net_layers.append(layers.Dense(1, kernel_regularizer=regularizers.l2(self.l2w), name='Output'))
            encoder = keras.Sequential(net_layers)
            self.encoders.append(encoder)


class CoxSENAM(CoxSENAMBase):
    def __init__(self, num_layers=4, num_nodes=16, act='relu', l2w=0.01, dropoutp=0.1, *args, **kwargs):

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.act = act
        self.dropoutp = dropoutp
        self.l2w = l2w

        super().__init__(*args, **kwargs)

    def _create_model(self):
        tf.random.set_seed(42)
        self.encoders = []
        for i in range(self.ds_input_shape):
            net_layers = [keras.Input(shape=(1,))]
            for i in range(self.num_layers):
                net_layers.append(
                    layers.Dense(self.num_nodes, activation=self.act, kernel_regularizer=regularizers.l2(self.l2w),
                                 name=f'layer{i}'))
                net_layers.append(layers.Dropout(self.dropoutp))
            net_layers.append(layers.Dense(1, kernel_regularizer=regularizers.l2(self.l2w), name='Output'))
            encoder = keras.Sequential(net_layers)
            self.encoders.append(encoder)


class CPH(CPHBase):
    def __init__(self, l2w=0.01, *args, **kwargs):
        self.l2w = l2w
        super().__init__(*args, **kwargs)

    def _create_model(self):
        tf.random.set_seed(42)
        self.encoder = keras.Sequential(
            [
                layers.Dense(1, name='Output', kernel_regularizer=regularizers.l2(self.l2w),
                             input_shape=(self.ds_input_shape,),
                             use_bias=False)
            ]
        )

# endregion derived models with hyperparameters
