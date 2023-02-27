import tensorflow as tf
from keras import layers
import numpy as np

num_actions = 2

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer():
    """
    The `Buffer` class implements Experience Replay.
    ---
    **Critic loss** - Mean Squared Error of `y - Q(s, a)`
    where `y` is the expected return as seen by the Target network,
    and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
    that the critic model tries to achieve; we make this target
    stable by updating the Target model slowly.
    **Actor loss** - This is computed using the mean of the value given by the Critic network
    for the actions taken by the Actor network. We seek to maximize this quantity.
    Hence we update the Actor network so that it produces actions that get
    the maximum predicted value as seen by the Critic, for a given state.
    """

    def __init__(self, num_states, buffer_capacity=10000, batch_size=50):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.reward_buffer_counter = 0
        self.reward_index = 0
        self.num_states = num_states

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((3*self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((3*self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 3))   # store three col of rewards
        self.next_state_buffer = np.zeros((3*self.buffer_capacity, self.num_states))
        
        self.actor_loss = None
        self.actor_grad = None
        self.critic_loss = None
        self.critic_grad = None

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.reward_buffer_counter % self.buffer_capacity
        
        # rewards batch has 3 columns
        if (self.reward_buffer_counter % 3 == 0 and self.reward_buffer_counter > 0):
            if self.reward_index+1 < self.buffer_capacity:
                self.reward_index += 1
                
            else:
                self.reward_index = 0
                self.reward_buffer_counter = 0
                index = 0
            
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[self.reward_index][self.reward_buffer_counter%3] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.reward_buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        target_actor,
        target_critic,
        actor_model,
        critic_model,
        critic_optimizer,
        actor_optimizer,
        gamma
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            # target actions: value network's predicted actions for next time step
            target_actions = target_actor(next_state_batch, training=True)

            # the following y is the TD Target
            # implement 3-step TD Target
            G3 = gamma*reward_batch[:,0] + gamma**2 * reward_batch[:,1] + gamma**3 * reward_batch[:,2]
            y = G3 + gamma**4 * target_critic(
                [next_state_batch, target_actions], training=True
            )

            # the critic value is the value given by the value network
            critic_value = critic_model([state_batch, action_batch], training=True)

            # this is the square of TD error
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            # collect critic's loss
            self.critic_loss = critic_loss
        
        # monitor critic's gradient
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        if len(critic_grad) > 0:
            self.critic_grad = 0.0
            for i in range(len(critic_grad)):
                self.critic_grad += tf.norm(critic_grad[i])

        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

            # collect the actor's loss, which should be increasing overtime
            self.actor_loss = actor_loss

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)

        # monitor actor's gradient
        if len(actor_grad) > 0:
            self.actor_grad = 0.0
            for i in range(len(actor_grad)):
                self.actor_grad += tf.norm(actor_grad[i])

        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, 
              target_actor,
              target_critic,
              actor_model,
              critic_model,
              critic_optimizer,
              actor_optimizer,
              gamma):
        # Get reward sampling range
        reward_record_range = min(self.reward_buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        reward_batch_indices = np.random.choice(reward_record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[3*reward_batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[3*reward_batch_indices])

        # sample 3 steps' rewards and state at 4th steps
        reward_batch = tf.convert_to_tensor(self.reward_buffer[reward_batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[3*reward_batch_indices+2])

        self.update(state_batch, action_batch,
                    reward_batch, next_state_batch,
                    target_actor,
                    target_critic,
                    actor_model,
                    critic_model,
                    critic_optimizer,
                    actor_optimizer,
                    gamma)


class DDPG_robo():
    def __init__(self, first_low=0.0, first_high=0.0, 
                 sec_low=0.0, sec_high=0.0, 
                 num_states=2.0, flag=""):
        """
        DDPG network for robot soccer.
        """
        # Learning rate for actor-critic models
        critic_lr = 0.05
        actor_lr = 0.05

        self.num_states = num_states
        self.flag = flag
        self.first_low = first_low
        self.first_high = first_high
        self.sec_low = sec_low
        self.sec_high = sec_high
        self.critic_optimizer = tf.keras.optimizers.Adadelta(critic_lr, clipnorm=1.0)
        self.actor_optimizer = tf.keras.optimizers.Adadelta(actor_lr, clipnorm=1.0)
        self.last_init = tf.keras.initializers.GlorotUniform(seed=120209)
        self.actor_model = self.get_actor(self.flag)   # is this training continuing on last checkpoint    
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor(self.flag)
        self.target_critic = self.get_critic()
        self.buffer = Buffer(num_states=num_states)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_actor_target(self, tau):
        for (a, b) in zip(self.target_actor.variables, self.actor_model.variables):
            a.assign(b * tau + a * (1 - tau))
            
    @tf.function
    def update_critic_target(self, tau):
        for (a, b) in zip(self.target_critic.variables, self.critic_model.variables):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self, flag):
        # attacker is trained with 3 layered NN
        if flag == "predict":
            inputs = layers.Input(shape=(self.num_states))
            out = layers.Dense(256, activation="relu", use_bias=True)(inputs)
            out = layers.Dense(256, activation="tanh", use_bias=True)(out)
            outputs = layers.Dense(num_actions, activation="tanh", use_bias=True, kernel_initializer=self.last_init)(out)

            max_num = np.array([self.first_low, self.first_high, self.sec_low, self.sec_low])
            outputs = outputs * np.amax(max_num)

        elif flag == "defender":
            leaky_relu = layers.LeakyReLU(alpha=0.3)
            inputs = layers.Input(shape=(self.num_states))
            out = layers.Dense(512, activation=leaky_relu, use_bias=True)(inputs)
            out = layers.Dense(512, activation=leaky_relu, use_bias=True)(out)
            outputs = layers.Dense(num_actions, activation="tanh", use_bias=True, kernel_initializer=self.last_init)(out)
            
        elif flag == "defender_predict":
            leaky_relu = layers.LeakyReLU(alpha=0.3)
            inputs = layers.Input(shape=(self.num_states))
            out = layers.Dense(256, activation=leaky_relu, use_bias=True)(inputs)
            out = layers.Dense(256, activation=leaky_relu, use_bias=True)(out)
            out = layers.Dense(256, activation=leaky_relu, use_bias=True)(out)
            out = layers.Dense(256, activation="tanh", use_bias=True)(out)
            out = layers.Dense(256, activation="tanh", use_bias=True)(out)
            outputs = layers.Dense(num_actions, activation="tanh", use_bias=True, kernel_initializer=self.last_init)(out)

        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # Le Cun weight initializer
        leaky_relu = layers.LeakyReLU(alpha=0.3)
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(128, activation=leaky_relu, use_bias=True)(state_input)
        state_out = layers.Dense(128, activation=leaky_relu, use_bias=True, kernel_initializer=self.last_init)(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(128, activation=leaky_relu, use_bias=True)(action_input)
        action_out = layers.Dense(128, activation=leaky_relu, use_bias=True, kernel_initializer=self.last_init)(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate(axis=1)([state_out, action_out])
        out = layers.Dense(256, activation=leaky_relu, use_bias=True)(concat)
        out = layers.Dense(256, activation=leaky_relu, use_bias=True, kernel_initializer=self.last_init)(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state, noise_object = None):
        """
        Return an action sampled by the actor model.
        """
        sampled_actions = tf.squeeze(self.actor_model(state))
        # reshape the sampled actions
        sampled_actions = tf.reshape(sampled_actions, [1,2])

        if noise_object is not None:
            noise = noise_object()
        else:
            noise = 0.

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        outputs = sampled_actions[0]
        outputs[0] = tf.clip_by_value(outputs[0] * self.first_high, self.first_low, self.first_high)   # kick power
        outputs[1] = tf.clip_by_value(outputs[1] * self.sec_high, self.sec_low, self.sec_high) # kick direction

        return np.squeeze(outputs)
