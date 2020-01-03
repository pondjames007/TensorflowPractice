"""
Core Tensorflow Concepts
"""

import tensorflow as tf
import numpy as np

# Build DAG
a = tf.constant([5, 3, 8])
b = tf.constant([3,-1, 2])
c = tf.add(a, b)

print(c) # Tensor("Add_7:0", shape=(3,), dtype=int32)


# Run
with tf.session() as sess:
    result = sess.run(c)
    print(result)   # [8, 2, 10]


# In Numpy
a = np.array([5, 3, 8])
b = np.array([3,-1, 2])
c = np.add(a, b)
print(c)    # [8, 2, 10]





# Visualize the Graph

# Perform in session
x = tf.constant([3,5,7], name='x')  # name the tensors and the operations
y = tf.constant([1,2,3], name='y')
z1 = tf.add(x, y, name='z1')
z2 = x * y
z3 = x2 - x1

with tf.session() as sess:
    # it will write the graph to the directory 'summaries'
    with tf.summary.FileWriter('summaries', sess.graph) as writer:
        a1, a3 = sess.run([z1, z3])

# Use Tensorboard to visualize
from google.datalab.ml import TensorBoard

TensorBoard().start('./summaries')





# Variables
def forward_pass(w, x):
    return tf.matmul(w, x)

def train_loop(x, niter=5):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        w = tf.get_variable(
            "weights", shape=(1,2),
            initializer=tf.truncated_normal_initializer(),
            trainable=True
        )
    preds = []
    for k in range(niter):
        preds.append(forward_pass(w, x))
        w = w + 0.1 # gradient update
    
    return preds

with tf.Session() as sess:
    preds = train_loop(tf.constant([[3.2, 5.1, 7.2], [4.3, 6.2, 8.3]])) # 2x3 matrix
    tf.global_variables_initializer().run()
    for i in range(len(preds)):
        print("{}:{}".format(i, preds[i].eval()))