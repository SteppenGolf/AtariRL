import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent  
from rl.policy import BoltzmannQPolicy  
from rl.memory import SequentialMemory


env = gym.make ('SpaceInvaders-v4', render_mode='rgb_array')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    
    while not done:

        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info, temp = env.step(action)
        score+=reward
    print('Iteration :{} Score :{}'.format(episode,score))
env.close()


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    
    return model

model = build_model(height, width, channels, actions)

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)



print(env)
print(env.observation_space)

agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()