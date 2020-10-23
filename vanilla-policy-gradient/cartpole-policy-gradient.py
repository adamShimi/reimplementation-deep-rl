# OpenAI Gym
import gym

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Define environment
env = gym.make('CartPole-v1')

nb_hidden = 50
nb_actions = env.action_space.n

model = keras.Sequential([
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_actions, activation="softmax", dtype = 'float64')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

episodes = 300
timer = 500

# Train for one step on the loss function
@tf.function
def training_step(batch):
  with tf.GradientTape() as tape:
    action_dist = model(batch)

    action = tf.random.categorical(action_dist,1)[0,0]
    action_prob = tf.math.log(action_dist[0, action])

  grad = tape.gradient(action_prob,model.trainable_weights)
  return (grad, action, action_dist)

for episode in range(episodes):
  grads = []
  rewards = []
  observation = env.reset()
  state = tf.constant([observation])

  # Sample an episode, while recording the data
  for t in range(timer):
    env.render()

    grad, action, action_dist = training_step(state)
    grads.append(grad)
    observation, reward, done, info = env.step(action.numpy())

    state = tf.constant([observation])

    rewards.append(reward)

    if done:
      time = t
      break

  print("Training Episode ", episode,"Return : ", time)
  discount = .95

  # Update the policy parameters
  for _  in range(time):
    return_episode = 0.0
    discount_factor = 1.0
    for reward in rewards:
      return_episode += reward*discount_factor
      discount_factor *= discount
    grad = list(map(lambda x: x*(-return_episode),grad))
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    grads = grads[1:]
    rewards = rewards[1:]

print("End of training")


nb_tests = 100
total_time = 0.0
# Test run
for test in range(nb_tests):
  observation = env.reset()
  for t in range(timer):
    state = tf.constant([observation])
    env.render()

    action_logits = model(state)
    action_dist = tf.nn.softmax(action_logits)
    action = tf.random.categorical(action_dist,1)[0,0]

    observation, reward, done, info = env.step(action.numpy())

    if done:
      time = t
      total_time+=time
      break
  print("Test Episode ", test, "Reward : ", time)

print("Average Time", total_time/nb_tests)

env.close()
