import random
import cPickle as pickle
import numpy as np
from evostra import EvolutionStrategy
from model import Model
import gym


class Agent:

    AGENT_HISTORY_LENGTH = 1
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000

    def __init__(self):
        self.env = gym.make('BipedalWalker-v2')
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION


    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction


    def load(self, filename='weights.pkl'):
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()


    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)


    def play(self, episodes, render=True):
        self.model.set_weights(self.es.weights)
        for episode in xrange(episodes):
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print "total reward:", total_reward


    def train(self, iterations):
        self.es.run(iterations, print_step=1)


    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in xrange(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward/self.EPS_AVG
