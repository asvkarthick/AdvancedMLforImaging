import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
import matplotlib

import os
import shutil

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import time, sys
import pickle
import timeit
from scipy.interpolate import make_interp_spline, BSpline
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Layer
from keras.optimizers import Adam
import keras.backend as K
import pandas as pd
import wandb
import string, random
import json

# os.environ["WANDB_API_KEY"] = "key_code"
os.environ["WANDB_API_KEY"] = "a7b3bca989f1cf6c97c2cbf57f77de63403a5fe5"

import numpy as np
from tensorflow import keras
import math
from tensorflow.keras import layers
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

plt.ioff()
import keras_cv
from keras.models import model_from_json
import keras

import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
print("Number of GPUs : ", len(gpus))

if gpus:
    gpu_memory = get_gpu_memory()
    maxmemory = max(gpu_memory)
    maxindex = gpu_memory.index(maxmemory)
    print("Available GPU memory : ", gpu_memory)
    print("Selecting GPU : ", maxindex)
    selected_gpu = gpus[maxindex]
    tf.config.set_visible_devices([selected_gpu], 'GPU')

# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#" * block + "-" * (barLength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = (
            embed_dim // num_heads
        )  # dimensionality of each attention head

        # create four dense layers (i.e., fully connected layers) that will be used in the attention mechanism.
        self.query_dense = DDense(embed_dim)
        self.key_dense = DDense(embed_dim)
        self.value_dense = DDense(embed_dim)
        self.combine_heads = DDense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, projection_dim]

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-4, **kwargs):
        self.eps = eps
        super(LayerNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=input_shape[-1:],
            initializer=tf.keras.initializers.Ones(),
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[-1:],
            initializer=tf.keras.initializers.Zeros(),
        )
        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class DBatch_Normalization(keras.layers.Layer):
    def __init__(self, var_epsilon):
        super(DBatch_Normalization, self).__init__()
        self.var_epsilon = var_epsilon

    def call(self, input_in):
        mean, variance = tf.nn.moments(input_in, [0, 1, 2])
        out = tf.nn.batch_normalization(
            input_in,
            mean,
            variance,
            offset=None,
            scale=None,
            variance_epsilon=self.var_epsilon,
        )
        return out


class Dsoftmax(keras.layers.Layer):
    def __init__(self):
        super(Dsoftmax, self).__init__()

    def call(self, input_in):
        out = tf.nn.softmax(input_in)
        return out


class DDDDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(DDDDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )
        super(DDDDense, self).build(input_shape)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class DDense(tf.keras.layers.Layer):
    def __init__(self, units=256):
        super(DDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.units),
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.zeros_initializer(),
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class DDropout(keras.layers.Layer):
    def __init__(self, drop_prop):
        super(DDropout, self).__init__()
        self.drop_prop = drop_prop

    def call(self, input_in, Training=None):
        if Training:
            out = tf.nn.dropout(input_in, rate=self.drop_prop)
        else:
            out = input_in
        return out


class DGeLU(tf.keras.layers.Layer):
    def __init__(self):
        super(DGeLU, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(input_shape[-1],), initializer="ones"
        )

    def call(self, input_in):
        return (
            0.5
            * input_in
            * (
                1
                + tf.math.erf(
                    input_in / (tf.math.sqrt(2.0) * tf.math.softplus(self.scale))
                )
            )
        )


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.dense1 = DDense(hidden_features)
        self.dense2 = DDense(out_features)
        self.dropout = DDropout(dropout_rate)
        self.gelu_1 = DGeLU()

    def call(self, x):
        x = self.dense1(x)
        x = self.gelu_1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.gelu_1(x)
        x = self.dropout(x)
        return x


class Dsoftmax(tf.keras.layers.Layer):
    def __init__(self):
        super(Dsoftmax, self).__init__()

    def call(self, input_in):
        out = tf.nn.softmax(input_in, axis=1)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, embed_dim, dropout)
        self.layernorm = LayerNorm(eps=1e-6)
        self.dropout = DDropout(dropout)

    def call(self, inputs, training=None):
        inputs_norm = self.layernorm(inputs)
        attn_output = self.att(inputs_norm, training=training)
        attn_output = self.dropout(attn_output, Training=training)
        out1 = attn_output + inputs
        out1_norm = self.layernorm(out1)
        mlp_output = self.mlp(out1_norm)
        return mlp_output + out1


class Deterministic_Conv(tf.keras.layers.Layer):
    def __init__(
        self, kernel_size, kernel_num, kernel_stride, padding="VALID", **kwargs
    ):
        super(Deterministic_Conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w",
            shape=(
                self.kernel_size,
                self.kernel_size,
                input_shape[-1],
                self.kernel_num,
            ),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
        )

    def call(self, input_in):
        out = tf.nn.conv2d(
            input_in,
            self.w,
            strides=[1, self.kernel_stride, self.kernel_stride, 1],
            padding=self.padding,
            data_format="NHWC",
        )
        return out


class DMaxPooling(tf.keras.layers.Layer):
    def __init__(self, pooling_size, pooling_stride, pooling_pad="SAME"):
        super(DMaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad

    def call(self, input_in):
        out, argmax_out = tf.nn.max_pool_with_argmax(
            input_in,
            ksize=[1, self.pooling_size, self.pooling_size, 1],
            strides=[1, self.pooling_stride, self.pooling_stride, 1],
            padding=self.pooling_pad,
        )
        return out


class DReLU(tf.keras.layers.Layer):
    def __init__(self):
        super(DReLU, self).__init__()

    def call(self, input_in):
        out = tf.nn.relu(input_in)
        return out


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class CustomCrossEntropyLoss(tf.keras.layers.Layer):
    def __init__(self, weight=None, name="custom_cross_entropy_loss"):
        super().__init__(name=name)
        self.weight = weight

    def call(self, y_true, y_pred):
        loss = -tf.reduce_sum(
            y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=-1
        )
        if self.weight is not None:
            loss *= self.weight
        return tf.reduce_mean(loss)


@keras.saving.register_keras_serializable(package="Deterministic_ViT")
class Deterministic_ViT(tf.keras.Model):
    def __init__(
        self,
        image_size=32,
        num_layers=12,
        kernel_size=3,
        kernel_num=128,
        kernel_stride=1,
        pooling_size=3,
        pooling_stride=2,
        pooling_pad="SAME",
        num_classes=8,
        embed_dim=256,
        num_heads=4,
        channels=3,
        dropout=0.1,
        var_epsilon=1e-4,
        **kwargs,
    ):
        super(Deterministic_ViT, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.num_classes = num_classes
        self.var_epsilon = var_epsilon
        self.dropout_rate = dropout

        self.conv_1 = Deterministic_Conv(
            kernel_size=self.kernel_size,
            kernel_num=self.kernel_num,
            kernel_stride=self.kernel_stride,
            padding="VALID",
        )
        self.conv_2 = Deterministic_Conv(
            kernel_size=self.kernel_size,
            kernel_num=self.kernel_num * 2,
            kernel_stride=self.kernel_stride,
            padding="VALID",
        )
        self.relu = DReLU()
        self.maxpooling = DMaxPooling(
            pooling_size=self.pooling_size,
            pooling_stride=self.pooling_stride,
            pooling_pad=self.pooling_pad,
        )
        self.zeropadding = tf.keras.layers.ZeroPadding2D(padding=1)
        self.enc_layers = [
            TransformerBlock(embed_dim, num_heads, embed_dim, dropout)
            for _ in range(num_layers)
        ]
        self.layernorm1 = LayerNorm(eps=1e-5)
        self.mysoftmax = Dsoftmax()
        self.ddense = DDense(units=1)
        self.final_dense = DDDDense(units=self.num_classes, activation="softmax")

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.zeropadding(x)
        x = self.maxpooling(x)
        x = tf.reshape(
            x, [-1, tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[-1]]
        )
        for layer in self.enc_layers:
            x = layer(x, training=training)
        x = self.layernorm1(x)
        x1 = self.ddense(x)
        xbar = self.mysoftmax(x1)
        weighted_representation = tf.matmul(xbar, x, transpose_a=True)
        weighted_representation = tf.squeeze(weighted_representation, -2)
        logits = self.final_dense(weighted_representation)
        return logits


# Preprocessing and augmentation functions
def preprocess(image, label):
    image = image / 255.0
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image, label


# Adds speckle noise (grainy pattern) to images
def add_speckle_noise(images, std_dev=0.1):
    noise = tf.random.normal(shape=tf.shape(images), mean=1.0, stddev=std_dev)
    noisy_images = images * noise
    return tf.clip_by_value(noisy_images, 0.0, 1.0)

# Adds scattering blur to images
def add_scattering_blur(images, kernel_size=3):
    channels = tf.shape(images)[-1]
    kernel = tf.ones((kernel_size, kernel_size, channels, 1)) / (kernel_size * kernel_size)
    blurred = tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.clip_by_value(blurred, 0.0, 1.0)

# Adds scattering occlusion to images
def add_scattering_occlusion(images, occlusion_prob=0.1, patch_size=5):
    def occlude_image(image):
        if tf.random.uniform([]) < occlusion_prob:
            h = tf.shape(image)[0]
            w = tf.shape(image)[1]
            c = tf.shape(image)[2]
            top = tf.random.uniform([], 0, h - patch_size + 1, dtype=tf.int32)
            left = tf.random.uniform([], 0, w - patch_size + 1, dtype=tf.int32)

            indices = []
            for i in range(patch_size):
                for j in range(patch_size):
                    indices.append([top + i, left + j])
            indices = tf.convert_to_tensor(indices, dtype=tf.int32)
            updates = tf.zeros((patch_size * patch_size, c), dtype=image.dtype)
            image = tf.tensor_scatter_nd_update(image, indices, updates)
        return image
    return tf.map_fn(occlude_image, images)

# Main function with added options for scattering attacks
def main_function(
    image_size=32,
    num_layers=12,
    num_classes=8,
    embed_dim=256,
    num_heads=4,
    channels=3,
    kernel_size=3,
    kernel_num=128,
    pooling_size=3,
    pooling_stride=2,
    kernel_stride=1,
    pooling_pad="SAME",
    dropout=0.1,
    batch_size=100,
    epochs=300,
    lr=0.001,
    lr_end=0.00001,
    Targeted=False,
    Random_noise=False,
    gaussain_noise_std=0.3,
    Adversarial_noise=False,
    epsilon=0.001,
    HCV=0.001,
    PGD_Adversarial_noise=False,
    Training=True,
    Testing=False,
    continue_training=False,
    maxAdvStep=20,
    stepSize=1,
    saved_model_epochs=10,
    # Scattering attack parameters
    Scattering_attack=False,
    scattering_type="speckle",  # Options: "speckle", "blur", "occlusion"
    scattering_std=0.1,         # for speckle noise
    scattering_kernel_size=3,   # for blur
    occlusion_prob=0.1,         # for occlusion
    occlusion_patch_size=5      # for occlusion
):
    PATH = "./mc_cct_saved_models/Deter_cct_epoch_{}_{}/".format(epochs, lr)
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    data_dir = "Padded_imgs"
    print(os.listdir(data_dir))

    image_shape = (image_size, image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_shape,
        batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_shape,
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    print("Class Names : ", class_names)
    print("epochs : ", epochs)
    print("batch size : ", batch_size)

    num_training_batches = tf.data.experimental.cardinality(train_ds).numpy()
    total_training_images = 0
    for imgs, lbls in train_ds:
        total_training_images += imgs.shape[0]
    print("Total training images : ", total_training_images)

    num_test_batches = tf.data.experimental.cardinality(test_ds).numpy()
    total_test_images = 0
    for imgs, lbls in test_ds:
        total_test_images += imgs.shape[0]
    print("Total test images : ", total_test_images)

    trans_model = Deterministic_ViT(
        image_size=image_size,
        num_layers=num_layers,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        channels=channels,
        kernel_size=kernel_size,
        kernel_num=kernel_num,
        kernel_stride=kernel_stride,
        pooling_size=pooling_size,
        pooling_stride=pooling_stride,
        pooling_pad=pooling_pad,
        dropout=dropout,
    )

    num_train_steps = epochs * int(total_training_images / batch_size)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr,
        decay_steps=num_train_steps,
        end_learning_rate=lr_end,
        power=2.0,
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate_fn)
    loss_fn = CustomCrossEntropyLoss()

    @tf.function
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            trans_model.trainable = True
            out = trans_model(x, training=True)
            loss = loss_fn(y, out)
        gradients = tape.gradient(loss, trans_model.trainable_weights)
        gradients = [
            grad if grad is not None else tf.zeros_like(var)
            for grad, var in zip(gradients, trans_model.trainable_weights)
        ]
        optimizer.apply_gradients(zip(gradients, trans_model.trainable_weights))
        return loss, out

    @tf.function
    def validation_on_batch(x, y):
        trans_model.trainable = False
        out = trans_model(x, training=True)
        total_vloss = loss_fn(y, out)
        return total_vloss, out

    @tf.function
    def test_on_batch(x, y, trans_model):
        out = trans_model(x, training=True)
        return out

    @tf.function
    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            trans_model.trainable = False
            prediction = trans_model(input_image, training=True)
            loss = loss_fn(input_label, prediction)
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad

    if Training:
        if os.path.exists(PATH):
            shutil.rmtree(PATH)
        os.makedirs(PATH)
        wandb.init(
            entity="bayestrans0", 
            project=f"kk_deter_mstar_epochs_{epochs}_layers_{num_layers}_lr_{lr}_batch_{batch_size}_embed_{embed_dim}_num_kernel_{kernel_num}_kernel_size_{kernel_size}_kernel_stride_{kernel_stride}_dropout_{dropout}"
        )
        if continue_training:
            saved_model_path = "./saved_models/cnn_epoch_{}/".format(saved_model_epochs)
            trans_model.load_weights(saved_model_path + "Deterministic_cnn_model")
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_err = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print("Epoch: ", epoch + 1, "/", epochs)
            acc1 = 0
            acc_valid1 = 0
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            acc_training = np.zeros(int(total_training_images / batch_size))
            err_training = np.zeros(int(total_training_images / batch_size))
            for step, (x, y) in enumerate(train_ds.take(batch_size)):
                y = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_training_images / batch_size))
                loss, out = train_on_batch(x, y)
                err1 += loss.numpy()
                corr = tf.equal(
                    tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1)
                )
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc1 += accuracy.numpy()
                if step % 100 == 0:
                    print("\n Step:", step, "Loss:", float(err1 / (tr_no_steps + 1.0)))
                    print(
                        "Total Training accuracy so far: %.3f"
                        % float(acc1 / (tr_no_steps + 1.0))
                    )
                tr_no_steps += 1
                wandb.log(
                    {
                        "Total Training Loss": loss.numpy(),
                        "Training Accuracy per minibatch": accuracy.numpy(),
                        "epoch": epoch,
                    }
                )
            train_acc[epoch] = acc1 / tr_no_steps
            train_err[epoch] = err1 / tr_no_steps
            print("Training Acc  ", train_acc[epoch])
            print("Training error", train_err[epoch])
            for step, (x, y) in enumerate(test_ds.take(batch_size)):
                y = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_test_images / batch_size))
                total_vloss, out = validation_on_batch(x, y)
                err_valid1 += total_vloss.numpy()
                corr = tf.equal(
                    tf.math.argmax(out, axis=-1), tf.math.argmax(y, axis=-1)
                )
                va_accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc_valid1 += va_accuracy.numpy()
                if step % 100 == 0:
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % va_accuracy)
                va_no_steps += 1
            valid_acc[epoch] = acc_valid1 / va_no_steps
            valid_err[epoch] = err_valid1 / va_no_steps
            stop = timeit.default_timer()
            if np.max(valid_acc) == valid_acc[epoch]:
                trans_model.save(PATH + "Deterministic_cnn_model_best.keras")
            stop = timeit.default_timer()
            wandb.log(
                {
                    "Average Training Loss": train_err[epoch],
                    "Average Training Accuracy": train_acc[epoch],
                    "Validation Loss": valid_err[epoch],
                    "Validation Accuracy": valid_acc[epoch],
                    "epoch": epoch,
                }
            )
            print("Total Training Time: ", stop - start)
            print(" Training Acc   ", train_acc[epoch])
            print(" Validation Acc ", valid_acc[epoch])
            print("------------------------------------")
        trans_model.save(PATH + "Deterministic_cnn_model_last.keras")
        with open(PATH + "training_validation_acc_error.pkl", "wb") as f1:
            pickle.dump([train_acc, valid_acc, train_err, valid_err], f1)
        with open(PATH + "Related_hyperparameters.txt", "w") as textfile:
            textfile.write(" Input Dimension : " + str(image_size))
            textfile.write("\n Kernel Number : " + str(kernel_num))
            textfile.write("\n Number of Classes : " + str(num_classes))
            textfile.write("\n No of epochs : " + str(epochs))
            textfile.write("\n Initial Learning rate : " + str(lr))
            textfile.write("\n Ending Learning rate : " + str(lr_end))
            textfile.write("\n kernel Size : " + str(kernel_size))
            textfile.write("\n Max pooling Size : " + str(pooling_size))
            textfile.write("\n Max pooling stride : " + str(pooling_stride))
            textfile.write("\n batch size : " + str(batch_size))
            textfile.write("\n---------------------------------")
            if Training:
                textfile.write("\n Total run time in sec : " + str(stop - start))
                if epochs == 1:
                    textfile.write("\n Averaged Training  Accuracy : " + str(train_acc))
                    textfile.write("\n Averaged Validation Accuracy : " + str(valid_acc))
                    textfile.write("\n Averaged Training  error : " + str(train_err))
                    textfile.write("\n Averaged Validation error : " + str(valid_err))
                else:
                    textfile.write("\n Averaged Training  Accuracy : " + str(np.mean(train_acc[epoch])))
                    textfile.write("\n Averaged Validation Accuracy : " + str(np.mean(valid_acc[epoch])))
                    textfile.write("\n Averaged Training  error : " + str(np.mean(train_err[epoch])))
                    textfile.write("\n Averaged Validation error : " + str(np.mean(valid_err[epoch])))
            textfile.write("\n---------------------------------")
            textfile.write("\n---------------------------------")
    if Testing:
        # ------------------- Scattering Attack Testing ------------------- #
        if Scattering_attack:
            test_path = "test_results_scattering_attack_{}_".format(scattering_type)
            if scattering_type == "speckle":
                test_path += "std_{}".format(scattering_std)
            elif scattering_type == "blur":
                test_path += "kernel_{}".format(scattering_kernel_size)
            elif scattering_type == "occlusion":
                test_path += "prob_{}_patch_{}".format(occlusion_prob, occlusion_patch_size)
            full_test_path = PATH + test_path + "/"
            if os.path.exists(full_test_path):
                shutil.rmtree(full_test_path)
            os.makedirs(full_test_path)
            trans_model = tf.keras.models.load_model(PATH + "Deterministic_cnn_model_best.keras")
            test_no_steps = 0
            acc_test = []
            for step, (x, y) in enumerate(test_ds.take(batch_size)):
                y_onehot = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_test_images / batch_size))
                if scattering_type == "speckle":
                    x_scatter = add_speckle_noise(x, std_dev=scattering_std)
                elif scattering_type == "blur":
                    x_scatter = add_scattering_blur(x, kernel_size=scattering_kernel_size)
                elif scattering_type == "occlusion":
                    x_scatter = add_scattering_occlusion(x, occlusion_prob=occlusion_prob, patch_size=occlusion_patch_size)
                else:
                    x_scatter = x
                predictions = test_on_batch(x_scatter, y_onehot, trans_model=trans_model)
                corr = tf.equal(tf.argmax(predictions, axis=-1), tf.argmax(y_onehot, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc_test.append(accuracy.numpy())
                test_no_steps += 1
            avg_accuracy = np.mean(acc_test)
            print("\nTest Accuracy with scattering attack ({}): {:.4f}".format(scattering_type, avg_accuracy))
            with open(full_test_path + "scattering_attack_results.txt", "w") as f:
                f.write("Scattering attack type: {}\n".format(scattering_type))
                if scattering_type == "speckle":
                    f.write("Speckle noise std: {}\n".format(scattering_std))
                elif scattering_type == "blur":
                    f.write("Blur kernel size: {}\n".format(scattering_kernel_size))
                elif scattering_type == "occlusion":
                    f.write("Occlusion probability: {}, patch size: {}\n".format(occlusion_prob, occlusion_patch_size))
                f.write("Test Accuracy: {}\n".format(avg_accuracy))
        # ------------------- Random Noise Testing ------------------- #
        elif Random_noise:
            test_path = "test_results_random_noise_{}/".format(gaussain_noise_std)
            full_test_path = PATH + test_path
            if os.path.exists(full_test_path):
                shutil.rmtree(full_test_path)
            os.makedirs(full_test_path)
            trans_model = tf.keras.models.load_model(PATH + "Deterministic_cnn_model_best.keras")
            test_no_steps = 0
            true_x = np.zeros(
                [int(total_test_images / batch_size), batch_size, image_size, image_size, channels]
            )
            true_y = np.zeros(
                [int(total_test_images / batch_size), batch_size, num_classes]
            )
            mu_out_ = np.zeros(
                [int(total_test_images / batch_size), batch_size, num_classes]
            )
            acc_test = np.zeros([int(total_test_images / batch_size)])
            for step, (x, y) in enumerate(test_ds.take(batch_size)):
                y = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_test_images / batch_size))
                true_x[test_no_steps, :, :, :, :] = x
                true_y[test_no_steps, :, :] = y
                noise = tf.random.normal(
                    shape=[batch_size, image_size, image_size, 1],
                    mean=0.0,
                    stddev=gaussain_noise_std,
                    dtype=x.dtype,
                )
                x = x + noise
                mu_out = test_on_batch(x, y, trans_model=trans_model)
                mu_out_[test_no_steps, :, :] = mu_out
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc_test[test_no_steps] = accuracy.numpy()
                if step % 100 == 0:
                    print("Total running accuracy so far: %.3f" % acc_test[test_no_steps])
                test_no_steps += 1
            test_acc = np.mean(acc_test)
            print("\nTest accuracy : ", test_acc)
            with open(full_test_path + "uncertainty_info.pkl", "wb") as pf:
                pickle.dump([mu_out_, true_x, true_y, test_acc], pf)
        # ------------------- Adversarial Noise Testing ------------------- #
        elif Adversarial_noise:
            if Targeted:
                test_path = "test_results_targeted_adversarial_noise_{}/".format(epsilon)
                full_test_path = PATH + test_path
                if os.path.exists(full_test_path):
                    shutil.rmtree(full_test_path)
                os.makedirs(full_test_path)
            else:
                test_path = "test_results_non_targeted_adversarial_noise_{}/".format(epsilon)
                full_test_path = PATH + test_path
                if os.path.exists(full_test_path):
                    shutil.rmtree(full_test_path)
                os.makedirs(full_test_path)
            trans_model = tf.keras.models.load_model(PATH + "Deterministic_cnn_model_best.keras")
            test_no_steps = 0
            true_x = np.zeros(
                [int(total_test_images / batch_size) + 1, batch_size, image_size, image_size, channels]
            )
            adv_perturbations = np.zeros(
                [int(total_test_images / batch_size), batch_size, image_size, image_size, channels]
            )
            true_y = np.zeros(
                [int(total_test_images / batch_size) + 1, batch_size, num_classes]
            )
            mu_out_ = np.zeros(
                [int(total_test_images / batch_size), batch_size, num_classes]
            )
            acc_test = np.zeros([int(total_test_images / batch_size)])
            for step, (x, y) in enumerate(test_ds.take(batch_size)):
                y = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_test_images / batch_size))
                true_x[test_no_steps, :, :, :, :] = x
                true_y[test_no_steps, :, :] = y
                if Targeted:
                    y_true_batch = np.zeros_like(y)
                    y_true_batch[:, adversary_target_cls] = 1.0
                    adv_perturbations1 = create_adversarial_pattern(x, y_true_batch)
                else:
                    adv_perturbations1 = create_adversarial_pattern(x, y)
                adv_x = x + epsilon * adv_perturbations1
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                adv_perturbations[test_no_steps, :, :, :, :] = adv_x
                mu_out = test_on_batch(adv_x, y, trans_model=trans_model)
                mu_out_[test_no_steps, :, :] = mu_out
                corr = tf.equal(tf.math.argmax(mu_out, axis=1), tf.math.argmax(y, axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc_test[test_no_steps] = accuracy.numpy()
                if step % 10 == 0:
                    print("Total running accuracy so far: %.3f" % accuracy.numpy())
                test_no_steps += 1
            test_acc = np.mean(acc_test)
            print("Adv Test accuracy : ", test_acc)
            with open(full_test_path + "uncertainty_info.pkl", "wb") as pf:
                pickle.dump([mu_out_, adv_perturbations, test_acc], pf)
        # ------------------- PGD Adversarial Noise Testing ------------------- #
        elif PGD_Adversarial_noise:
            if Targeted:
                test_path = "test_results_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/".format(
                    epsilon, maxAdvStep, stepSize
                )
                full_test_path = PATH + test_path
                if os.path.exists(full_test_path):
                    shutil.rmtree(full_test_path)
                os.makedirs(full_test_path)
            else:
                test_path = "test_results_non_targeted_PGDadversarial_noise_{}/".format(epsilon)
                full_test_path = PATH + test_path
                if os.path.exists(full_test_path):
                    shutil.rmtree(full_test_path)
                os.makedirs(full_test_path)
            trans_model = tf.keras.models.load_model(PATH + "Deterministic_cnn_model_best.keras")
            trans_model.trainable = False
            test_no_steps = 0
            true_x = np.zeros(
                [int(total_test_images / batch_size), batch_size, image_size, image_size, channels]
            )
            adv_perturbations = np.zeros(
                [int(total_test_images / batch_size), batch_size, image_size, image_size, channels]
            )
            true_y = np.zeros(
                [int(total_test_images / batch_size), batch_size, num_classes]
            )
            mu_out_ = np.zeros(
                [int(total_test_images / batch_size), batch_size, num_classes]
            )
            acc_test = np.zeros([int(total_test_images / batch_size)])
            for step, (x, y) in enumerate(test_ds.take(batch_size)):
                y = tf.one_hot(np.squeeze(y).astype(np.float32), depth=num_classes)
                update_progress(step / int(total_test_images / batch_size))
                true_x[test_no_steps, :, :, :, :] = x
                true_y[test_no_steps, :, :] = y
                adv_x = x + tf.random.uniform(x.shape, minval=-epsilon, maxval=epsilon)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                for advStep in range(maxAdvStep):
                    if Targeted:
                        y_true_batch = np.zeros_like(y)
                        y_true_batch[:, adversary_target_cls] = 1.0
                        adv_perturbations_ = create_adversarial_pattern(adv_x, y_true_batch)
                    else:
                        adv_perturbations_ = create_adversarial_pattern(adv_x, y)
                    adv_x = adv_x + stepSize * adv_perturbations_
                    adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                    adv_perturbations[test_no_steps, :, :, :, :] = adv_x
                mu_out = test_on_batch(adv_x, y, trans_model=trans_model)
                mu_out_[test_no_steps, :, :] = mu_out
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
                acc_test[test_no_steps] = accuracy.numpy()
                if step % 50 == 0:
                    print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
                test_no_steps += 1
            test_acc = np.mean(acc_test)
            print("PGD Test accuracy : ", test_acc)
            with open(full_test_path + "uncertainty_info.pkl", "wb") as pf:
                pickle.dump([mu_out_, true_x, true_y, adv_perturbations, test_acc], pf)

if __name__ == "__main2__":
    main_function()

if __name__ == "__main__":
    main_function()
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)                  
    main_function(Random_noise=True, gaussain_noise_std=0.001, epsilon=0.005, Training=False, Testing=True,
                  Adversarial_noise=False, PGD_Adversarial_noise=False)                 
    main_function(Random_noise=True, gaussain_noise_std=0.01, epsilon=0.01, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.05, epsilon=0.05, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.1, epsilon=0.07, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.2, epsilon=0.08, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.3, epsilon=0.1, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.4, epsilon=0.15, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.5, epsilon=0.2, Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    ##################################################################################################                  
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)                 
    main_function(Random_noise=False, gaussain_noise_std=0.001, epsilon=0.005, Training=False, Testing=False,
                  Adversarial_noise=True, PGD_Adversarial_noise=False)               
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.01, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.05, epsilon=0.05, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.1, epsilon=0.07, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.2, epsilon=0.08, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.3, epsilon=0.1, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.4, epsilon=0.15, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.5, epsilon=0.2, Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    ####################################################################################################                   
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)                
    main_function(Random_noise=False, gaussain_noise_std=0.001, epsilon=0.005, Training=False, Testing=False,
                  Adversarial_noise=False, PGD_Adversarial_noise=True)                  
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.01, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.05, epsilon=0.05, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.1, epsilon=0.07, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.2, epsilon=0.08, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.3, epsilon=0.1, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.4, epsilon=0.15, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.5, epsilon=0.2, Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
####################################################################################################                   
    # Scattering attack tests
    main_function(Training=False, Testing=True,
                  Scattering_attack=True, scattering_type="speckle", scattering_std=0.1)
    main_function(Training=False, Testing=True,
                  Scattering_attack=True, scattering_type="blur", scattering_kernel_size=5)
    main_function(Training=False, Testing=True,
                  Scattering_attack=True, scattering_type="occlusion", occlusion_prob=0.2, occlusion_patch_size=7)
