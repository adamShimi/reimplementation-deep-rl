# OpenAI Gym
import gym

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
from numpy import random
from datetime import datetime

# Define environment
seed = 1
env = gym.make('CartPole-v1')
env.seed(seed)
env.action_space.seed(seed)

rng = np.random.default_rng()

# Fix hyperparameters
nb_hidden = 30
nb_actions = env.action_space.n

learning_rate = 1e-3
episodes = 200
timer = 1000
gamma = .95
size_experience = 50
batch_size = 10
epsilon = 0.1


experience = []
priorities = []
probabilities = []

model = keras.Sequential([
  keras.layers.Dense(nb_hidden, activation="tanh", dtype='float64'),
  keras.layers.Dense(nb_hidden, activation="tanh", dtype='float64'),
  keras.layers.Dense(nb_actions, dtype = 'float64')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

# Train for one step on the loss function
@tf.function
def training_step(batch):
  targets = batch[:,0]
  actions = tf.expand_dims(tf.dtypes.cast(batch[:,1],tf.int32),1)
  states = batch[:,2:]
  with tf.GradientTape() as tape:
    action_values = model(states)
    chosen_action_values = tf.gather_nd(action_values, actions, batch_dims=1)
    loss = tf.math.pow((targets - chosen_action_values),2)
  grad = tape.gradient(loss,model.trainable_weights)
  optimizer.apply_gradients(zip(grad, model.trainable_weights))

def choose_action(state,explore):
  explore_or_exploit = rng.binomial(1,1-epsilon)
  if explore_or_exploit == 0:
    action = env.action_space.sample()
  else:
    action_values = model(state)
    max_val = tf.reduce_max(action_values[0])
    action = random.choice([i for (i,j) in enumerate(action_values[0].numpy()) if j == max_val.numpy()])
  return action

def compute_priority(state,next_state,action,reward):
  action_values = model(state)
  q_value = action_values[0][action]
  action_values = model(next_state)
  max_val = tf.reduce_max(action_values[0])
  return abs(reward + gamma*max_val - q_value)

def compute_probabilities(prios):
  total = sum(prios)
  return list(map(lambda x : x / total,prios))

for episode in range(episodes):
  observation = env.reset()
  state = tf.constant([observation])

  # Sample an episode, while recording the data
  for t in range(timer):
    env.render()

    action = choose_action(state,True)
    observation, reward, done, info = env.step(action)

    next_state = tf.constant([observation])
    experience.append((state,action,reward,next_state,done))
    priorities.append(compute_priority(state,next_state,action,reward))
    if len(experience) > size_experience:
      experience.pop(0)
      priorities.pop(0)

    probabilities = compute_probabilities(priorities)

    state = next_state

    # Update the q-value network
    # Sample batch_size transitions from experience
    sample = random.choice(len(experience), batch_size, p=probabilities)
    batch = []
    for i in sample:
      rand_state, rand_action, rand_reward, rand_next_state,rand_done = experience[i]
      priorities[i] = compute_priority(rand_state, rand_next_state, rand_action, rand_reward)
      if rand_done:
        rand_target = rand_reward
        rand_target = tf.reshape(tf.constant(np.float64(rand_target)),(1,1))
        rand_action = tf.reshape(tf.constant(np.float64(rand_action)),(1,1))
      else:
        rand_action_values = model(rand_next_state)
        rand_max_val = tf.reduce_max(rand_action_values[0])
        rand_target = rand_reward + gamma*rand_max_val
        rand_target = tf.reshape(tf.constant(rand_target),(1,1))

        rand_action = tf.reshape(tf.constant(np.float64(rand_action)),(1,1))

      batch.append(tf.concat([rand_target,rand_action,rand_state],1))

    probabilities = compute_probabilities(priorities)

    batch = tf.concat(batch, 0)
    training_step(batch)

    if done:
      time = t
      break

  print("Training Episode ", episode,"Return : ", time)
  tf.summary.scalar('Length of Episode', data=time, step=episode)

print("End of training")


nb_tests = 100
total_time = 0.0
# Test run
for test in range(nb_tests):
  observation = env.reset()
  for t in range(timer):
    state = tf.constant([observation])
    env.render()

    action = choose_action(state,False)
    observation, reward, done, info = env.step(action)

    if done:
      time = t
      total_time+=time
      break
  print("Test Episode ", test, "Reward : ", time)

print("Average Time", total_time/nb_tests)

env.close()
