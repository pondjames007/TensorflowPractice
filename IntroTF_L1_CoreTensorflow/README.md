# Lesson 1: Core Tensorflow

## Lazy Evaluation
The spirit of Tensorflow coding is **Lazy Evaluation**, it builds **DAG (Direct Acyclic Graph)**.
All operations only build DAG, it doesn't execute it.
You need to run it as part of what is called **session** (return *np array*)
* It can be imagined as writing a program then compile
* You need to **run** the **graph** to get the result
* **tf.eager** can let you get the result directly, but it is not good in production
    * from tensorflow.contrib.eager.python import tfe
    * tfe.enable_eager_execution()
```
    # Build DAG
    a = tf.constant([5, 3, 8])
    b = tf.constant([3,-1, 2])
    c = tf.add(a, b)

    print(c)    # Tensor("Add_7:0", shape=(3,), dtype=int32)


    # Run
    with tf.session() as sess:
        result = sess.run(c)
        print(result)   # [8, 2, 10]

    
    # In Numpy
    a = np.array([5, 3, 8])
    b = np.array([3,-1, 2])
    c = np.add(a, b)
    print(c)    # [8, 2, 10]
```

## DAG (Direct Acycic Graph)
* Nodes: operations on tensors
* Edges: Tensors (N-dimensional Array)(Data)
* Tensorflow can optimize the graph by merging successive nodes where necessary
* Tensorflow can insert send and receive nodes to distribute the graph across machines

## Session
* The session class represents the connection between the Python program we write, and the C++ runtime
* The session object provides access to the devices on the local machine, and to remote devices using the distributor Tensorflow runtime
```
    with tf.session() as sess:
        result = sess.run(z)    # both are the same
        result = z.eval()
```
* tensor.eval() is a **shortcut** of sess.run(tensor)
* sess.run() can pass in **a list of tensors** to evaluate

## Visualize the Graph
* Perform in session
```
    x = tf.constant([3,5,7], name='x')  # name the tensors and the operations
    y = tf.constant([1,2,3], name='y')
    z1 = tf.add(x, y, name='z1')
    z2 = x * y
    z3 = x2 - z1

    with tf.session() as sess:
        with tf.summary.FileWriter('summaries', sess.graph) as writer:  
        # it will write the graph to the directory 'summaries'
            a1, a3 = sess.run([z1, z3])
```
* It will output a binary file in your designate directory
* Use **Tensorboard** to visualze it
```
    from google.datalab.ml import TensorBoard

    TensorBoard().start('./summaries')
```

## Variables
A variable is a tensor whose value is initalized and then typically changed as the program runs (應該是為了往後表示NN的weights)
```
	def forward_pass(w, x):
		return tf.matmul(w, x)

	def train_loop(x, niter=5):
		with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
			w = tf.get_variable(
					"weights", shape=(1, 2),
					initializer=tf.truncated_normal_initializer(),
					trainable=True)
		pred = []
		for k in range(niter):
			preds.append(forward_pass(w, x))
			w = w + 0.1		# gradient update
		return preds


	with tf.Session() as sess:
		preds = train_loop(tf.constant([[3.2, 5.1, 7.2], [4.3, 6.2, 8.3]]))	# 2x3 matrix
		tf.global_variables_initailizer().run()
		for i in range(len(preds)):
			print("{}:{}".format(i, preds[i].eval()))
```
* truncated_normal_initializer():
    * initialize a variable in **Gaussian Distribution** (with truncation)
* trainable:
    * it is trainable while training
* tf.get_variable is better than using tf.variable()
* You need to initialize variables in **Session**
    * tf.global_variables_initializer().run() -> initialize all variables


### Scope
A place to tell Tensorflow to **reuse** the variable each time instead of creating new variables each time.
* If we call train_loop again, the variable will resume from where they left off

### Placeholder
* Allows you to feed in values, such as reading from a file
```
	a = tf.placeholder("float", None)
	b = a * 4
	print(a)		# Tensor("Placeholder", dtype=float32)
	
	with tf.Session() as sess:
		print(sess.run(b, feed_dict={a: [1,2,3]})
```


LAB: *training-data-analyst > courses > machine_learning > deepdive > 03_tensorflow > labs*and open*a_tfstart.ipynb*

## Debugging
### Reshape a tensor
* tf.reshape()
* tf.expand_dims(tensor, axis): inserts a dimension of 1 into a tensor’s shape
* tf.slice(tensor, start_point, num_rows_cols)
* tf.squeeze(): remove a dimension of q from a tensor’s shape

### Change data type
* tf.cast(tensor, new_type)

### Debug full program
* tf.Print(): print out tensor values when meet some conditions
* tfdbg: debugger run in the terminal
* Tensorboard: higher level debugging for a NN

The default level of logging is WARN
* You can change the level
	* tf.logging.set_verbosity(tf.logging.INFO)
	* debug, info, warn, error, fatal
	* INFO is good for development
	* WARN is good for production