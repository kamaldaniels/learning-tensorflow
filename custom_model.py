import tensorflow as tf
import numpy as np


def model(features, labels, mode):
    # build linear model and predict values
    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square(y - labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    global_step = tf.train.get_global_step()
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


# data
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x': x}, y, batch_size=4, num_epochs=1000)

# train
estimator = tf.contrib.learn.Estimator(model_fn=model)
estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn, steps=10))
