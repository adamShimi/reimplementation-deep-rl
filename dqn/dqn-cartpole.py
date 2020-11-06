# OpenAI Gym
import gym

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import random as random

# Define environment
env = gym.make('CartPole-v1')
env.seed(1)

rng = np.random.default_rng()

# Fix hyperparameters
nb_hidden = 30
nb_actions = env.action_space.n

learning_rate = 1e-3
episodes = 300
timer = 500
gamma = .95
size_experience = 50
batch_size = 10
epsilon = 0.1

model = keras.Sequential([
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_actions, dtype = 'float64')
])

optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train for one step on the loss function
def training_step(batch):
  for (state, action, target) in batch:
    with tf.GradientTape() as tape:
      action_values = model(state)
      loss = pow((target - action_values[0][action]),2)
    grad = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))

def choose_action(state):
  explore_or_exploit = rng.binomial(1,1-epsilon)
  if explore_or_exploit == 1:
    action_values = model(state)
    max_val = tf.reduce_max(action_values[0])
    action = random.choice([i for (i,j) in enumerate(action_values[0].numpy()) if j == max_val.numpy()])
  else:
    action = env.action_space.sample()
  return action

experience = []
for episode in range(episodes):
  observation = env.reset()
  state = tf.constant([observation])

  # Sample an episode, while recording the data
  for t in range(timer):
    env.render()

    action = choose_action(state)
    observation, reward, done, info = env.step(action)

    next_state = tf.constant([observation])
    experience.append((state,action,reward,next_state,done))
    if len(experience) > size_experience:
      experience.pop(0)

    state = next_state

    # Update the q-value network
    # Sample batch_size transitions from experience
    batch = []
    for _ in range(batch_size):
      rand_state, rand_action, rand_reward, rand_next_state,rand_done = random.choice(experience)
      if rand_done:
        rand_target = rand_reward
      else:
        rand_action_values = model(rand_next_state)
        rand_max_val = max(rand_action_values)
        rand_target = rand_reward + gamma*rand_max_val
      batch.append((rand_state,rand_action,rand_target))
    training_step(batch)

    if done:
      time = t
      break

  print("Training Episode ", episode,"Return : ", time)

print("End of training")


nb_tests = 100
total_time = 0.0
# Test run
for test in range(nb_tests):
  observation = env.reset()
  for t in range(timer):
    state = tf.constant([observation])
    env.render()

    action = choose_action(state)
    observation, reward, done, info = env.step(action)

    if done:
      time = t
      total_time+=time
      break
  print("Test Episode ", test, "Reward : ", time)

print("Average Time", total_time/nb_tests)

env.close()
