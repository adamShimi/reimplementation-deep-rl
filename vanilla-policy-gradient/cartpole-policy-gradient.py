# OpenAI Gym
import gym

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Define environment
env = gym.make('CartPole-v1')
env.seed(1)

# Fix hyperparameters
nb_hidden = 30
nb_actions = env.action_space.n

learning_rate = 1e-5
episodes = 300
timer = 500
discount = 1.0

model = keras.Sequential([
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_hidden, activation="relu", dtype='float64'),
  keras.layers.Dense(nb_actions, activation="softmax", dtype = 'float64')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train for one step on the loss function
@tf.function
def training_step(batch):
  with tf.GradientTape() as tape:
    action_dist = model(batch)

    action = tf.random.categorical(action_dist,1)[0,0]
    action_prob = -tf.math.log(action_dist[0, action])

  grad = tape.gradient(action_prob,model.trainable_weights)
  return (grad, action, action_prob)

for episode in range(episodes):
  grads = []
  rewards = []
  action_probs = []
  observation = env.reset()
  state = tf.constant([observation])

  # Sample an episode, while recording the data
  for t in range(timer):
    env.render()

    grad, action, action_prob = training_step(state)
    grads.append(grad)
    action_probs.append(action_prob)
    observation, reward, done, info = env.step(action.numpy())

    state = tf.constant([observation])

    rewards.append(reward)

    if done:
      time = t
      break


  # Update the policy parameters
  loss_episode = 0.0
  for _  in range(time):
    return_episode = 0.0
    discount_factor = 1.0
    for reward in rewards:
      return_episode += reward*discount_factor
      discount_factor *= discount
    loss_episode += return_episode*action_probs[0]
    grad = list(map(lambda x: x*return_episode,grads[0]))
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    grads = grads[1:]
    rewards = rewards[1:]
    print("Action_prob: ",action_probs[0].numpy())
    action_probs = action_probs[1:]

  print("Training Episode ", episode,"Return : ", time, "Loss : ", loss_episode.numpy())

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
