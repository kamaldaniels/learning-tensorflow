import tensorflow as tf

session = tf.Session()

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
session.run(init)

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print('org loss:', session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# the best W and b
fix_W = tf.assign(W, [-1.])
fix_b = tf.assign(b, [1.])
session.run([fix_W, fix_b])
print('manual fix loss:', session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# reset values to wrong
session.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

print('training...')

for i in range(1000):
    session.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = session.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
