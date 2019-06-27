import tensorflow as tf

feautres = tf.constant([1., 2., 3., 4., 5., 6., 7., 8., 9.])
feautres = tf.constant([10., 11., 12., 13., 14., 15., 16., 17., 18.])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(feautres))
print(sess.run(feautres[:, tf.newaxis]))