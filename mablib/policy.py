import random, numpy

class Policy:
    def __init__(self, size, seed=42):
        self.size = size
        self.rnds = numpy.random.mtrand.RandomState(seed)
        self.rewards = {}
        
        for arm in range(size):
            self.rewards[arm] = []
    
    def act(self):
        pass
    
    def update(self, arm, reward):
        self.rewards[arm].append(reward)
    
    def reduce(self, f=numpy.mean):
        return numpy.array([f(self.rewards[i]) for i in range(self.size)])
    
    def reward(self, probs):
        return self.rnds.choice([1, 0], p=[probs, 1.0 - probs])

class RandomPolicy(Policy):
    def __init__(self, size, seed=42):
        super().__init__(size, seed)
    
    def act(self):
        return self.rnds.randint(0, self.size)

class EpsilonGreedyPolicy(Policy):
    def __init__(self, size, seed=42, epsilon=0.1):
        super().__init__(size, seed)
        self.epsilon = epsilon
    
    def act(self):
        if self.rnds.choice([1, 0], p=[self.epsilon, 1.0 - self.epsilon]):
            return self.rnds.randint(0, self.size)
        else:
            return numpy.argmax(self.reduce())

class ThompsonSamplingPolicy(Policy):
    def __init__(self, size, seed=42):
        super().__init__(size, seed)
        self.alpha = numpy.ones(size)
        self.beta = numpy.ones(size)
    
    def act(self):
        prior = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(self.size)]
        return numpy.argmax(prior)
    
    def update(self, arm, reward):
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
