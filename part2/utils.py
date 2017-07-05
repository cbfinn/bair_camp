import tensorflow as tf

class Robot(object):
  def __init__(self, env):
    self.dim_action = self.env.action_space.n
    self.dim_obs = self.env.observation_space.shape[0]
    self.session = tf.InteractiveSession()
    self.initialize_network()

  def initialize_network(self):
    self.obs = tf.placeholder(tf.float32)
    self.action_label = tf.placeholder(tf.float32)
    self.output = None # TODO - make network
    self.session.run(tf.global_variables_initializer())

  def _set_loss(self, loss_expr):
    self.loss = loss_expr
    error = loss_expr(self.action_label, self.output)
    self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)

  def get_action(self, obs):
    return sess.run(self.output, feed_dict={self.obs: obs})

  def _train_step(self, obs, actions):
    feed_dict = {self.obs: obs,
                 self.action_label: actions}
    self.session.run(self.train_op, feed_dict=feed_dict)

