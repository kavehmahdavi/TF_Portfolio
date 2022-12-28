"""
This uses Keras to:
    Load a prebuilt MNIST dataset.
    Build a neural network machine learning model that classifies images.
    Train this neural network.
    Evaluate the accuracy of the model

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 14/12/2022
"""
import os
import io
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import threading
from sklearn.utils import shuffle

# Disable Tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def plot_cm(cm, _plot=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array 2d): confusion matrix
       _plot (bool): If True, it plots the graph.

    Return:
        A figure object
    """
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    if _plot:
        plt.show()
    else:
        return fig


def fig2png(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    Args:
        figure (object): matplotlib plot specified by 'figure'


    Return:
        A png image
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the test_images.
    _probability_model = tf.keras.Sequential([model,
                                              tf.keras.layers.Softmax()])
    predicted = _probability_model.predict(x_test)
    predicted = np.argmax(predicted, axis=1)

    # Calculate the confusion matrix using sklearn.metrics
    cm = confusion_matrix(y_target=y_test,
                          y_predicted=predicted)

    figure = plot_cm(cm)
    cm_image = fig2png(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def lunch_tb(_log_address):
    """

    Args:
        _log_address:
    """
    tb_thread = threading.Thread(target=lambda: os.system('tensorboard --logdir={}'.format(_log_address)),
                                 daemon=True)
    tb_thread.start()


# Load and prepare the MNIST dataset.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# shuffle data
x_train, y_train = shuffle(x_train, y_train, random_state=4)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Plotting the MNIST dataset
if False:
    for i in range(4):
        plt.subplot(220 + 1 + i)
        plt.imshow(x_train[i + 100], cmap=plt.get_cmap('gray'))
    plt.show()

log_address = '../outputs/image_classification/logs/'
shutil.rmtree(log_address + 'image/')
logdir = log_address + "image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

# Build a machine learning mode
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model.fit(x_train,
          y_train,
          epochs=6,
          verbose=1,
          callbacks=[tensorboard_callback, cm_callback],
          validation_data=(x_test, y_test))

# Lunch TensorBoard
lunch_tb(log_address)
