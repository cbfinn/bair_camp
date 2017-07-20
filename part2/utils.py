import numpy as np
import pickle
import tensorflow as tf

from PIL import Image

class Robot(object):
  def __init__(self, env, dim_action=None, dim_obs=None):
    if dim_action is None:
        self.dim_action = self.env.action_space.shape[0]
    else:
        self.dim_action = dim_action
    if dim_obs is None:
        self.dim_obs = self.env.observation_space.shape[0]
    else:
        self.dim_obs = dim_obs
    tf.reset_default_graph()
    self.session = tf.InteractiveSession()
    self.initialize_network()

  def initialize_network(self, dim_hiddens = (40, 40)):
    self.obs = tf.placeholder(tf.float32)
    self.action_label = tf.placeholder(tf.float32)

    # make weight matrices and biases
    cur_dim = self.dim_obs
    cur_vec = tf.reshape(self.obs, [-1, cur_dim])
    bias_init = tf.constant_initializer(0.1)
    fc_init = tf.truncated_normal_initializer(stddev=0.01)
    for i in range(len(dim_hiddens)):
        w = tf.get_variable('w'+str(i), [cur_dim, dim_hiddens[i]], initializer=fc_init)
        b = tf.get_variable('b'+str(i), [dim_hiddens[i]], initializer=bias_init)
        cur_vec = tf.nn.relu(tf.matmul(cur_vec, w) + b)
        cur_dim = dim_hiddens[i]
    w = tf.get_variable('wout', [dim_hiddens[-1], self.dim_action], initializer=fc_init)
    b = tf.get_variable('bout', [self.dim_action], initializer=bias_init)
    self.output = tf.matmul(cur_vec, w) + b

    self.session.run(tf.global_variables_initializer())

  def _set_loss(self, loss_expr):
    action_label = tf.reshape(self.action_label, [-1, self.dim_action])
    self.loss = loss_expr
    self.error = error = loss_expr(action_label, self.output)
    self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)
    # initialize Adam variables by re-initializing all variables.
    self.session.run(tf.global_variables_initializer())

  def get_image(self):
    self.env.render()
    image = self.env.env.viewer.get_image()
    pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
    return np.flipud(np.array(pil_image))

  def get_action(self, obs):
    obs = np.reshape(obs, (1, -1))
    return self.session.run(self.output, feed_dict={self.obs: obs})

  def _train_step(self, obs, actions):
    batch_size = 32
    batch_idxs = np.random.randint(0, self.num_demos*self.max_timesteps, [batch_size])
    feed_dict = {self.obs: obs[batch_idxs],
                 self.action_label: actions[batch_idxs]}
    error, _ = self.session.run([self.error, self.train_op], feed_dict=feed_dict)
    return error

  def _collect_demonstrations(self, num_demos=50):
    # used to generate demonstration pkl files
    expert_policy = load_policy.load_policy('experts/' + self.env.spec.id + '.pkl', self.session)
    tf_util.initialize(self.session)
    observations = []
    actions = []
    videos = []
    for i in range(num_demos):
      print(i)
      obs = self.env.reset()
      for t in range(self.max_timesteps):
        # only store videos for 2 trajectories.
        if len(videos) < self.max_timesteps * 2:
          videos.append(self.get_image())
        action = expert_policy(obs[None,:])
        observations.append(obs)
        actions.append(action)
        obs, _, _, _ = self.env.step(action)
    self.expert_video_clip = mpy.ImageSequenceClip(videos, fps=20*2)
    self.expert_data = {'observations': np.array(observations),
                        'actions': np.array(actions),
                        'video': videos}
    with open('experts/' + self.name + '_demos.pkl', 'wb') as f:
        pickle.dump(self.expert_data, f)


