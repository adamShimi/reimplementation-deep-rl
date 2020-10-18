# OpenAI Gym
import gym

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Define environment
env = gym.make('FrozenLake-v0')

nb_hidden = 30
nb_actions = env.action_space.n

model = keras.Sequential([
  keras.layers.Dense(nb_hidden),
  keras.layers.Dense(nb_hidden),
  keras.layers.Dense(nb_actions)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

episodes = 1000
timer = 100

for _ in range(episodes):
  grads = []
  rewards = []
  observation = env.reset()
  state = tf.constant(observation)
  state = tf.expand_dims(state,0)
  state = tf.expand_dims(state,0)

  # Sample an episode, while recording the data
  for _ in range(timer):
    env.render()

    with tf.GradientTape() as tape:
      action_logits = model(state)

      action = tf.random.categorical(action_logits,1)[0,0]
      action_dist = tf.nn.softmax(action_logits)
      action_prob = -tf.math.log(action_dist[0, action])

    grads.insert(0,tape.gradient(action_prob,model.trainable_weights))

    observation, reward, done, info = env.step(action.numpy())

    state = tf.constant(observation)
    state = tf.expand_dims(state,0)

    if done:
      if reward == 0:
        rewards.insert(0,-1)
      else:
        rewards.insert(0,reward)

      break
    else:
      rewards.insert(0,reward)

  return_episode = 0.0
  # Update the policy parameters
  for (grad,reward)  in zip(grads, rewards):
    return_episode += reward
    grad = list(map(lambda x: x*return_episode,grad))
    optimizer.apply_gradients(zip(grad, model.trainable_weights))

print("End of training")


nb_tests = 100
total_reward = 0
# Test run
for _ in range(nb_tests):
  for _ in range(timer):
    env.render()
    state = tf.expand_dims(tf.constant(observation),0)

    action_logits = model(state)
    action = tf.random.categorical(action_dist,1)[0,0]

    observation, reward, done, info = env.step(action.numpy())

    if done:
      total_reward+=reward
      break

print(total_reward/nb_tests)

env.close()
