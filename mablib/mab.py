import numpy

class MultiArmedBandit:
    def __init__(self, views, arms, policy, verbose=False):
        self.views = views
        self.arms = arms
        self.policy = policy
        self.verbose = verbose
        
    def run(self):
        
        self.arms_visits = []
        self.arms_reward = []
        
        # Reset arms
        for arm in range(len(self.arms)):
            self.policy.update(arm, 0.0)

        # Run episodes
        for view in range(self.views):
            arm = self.policy.act()

            # Replace reward to sales or no-sales.
            reward = self.policy.reward(self.arms[arm])

            self.policy.update(arm, reward)
            
            # Save states
            self.arms_visits.append(arm)
            self.arms_reward.append(reward)

            if self.verbose:
                print('{}: arm {} with reward {}'.format(view, arm, reward))

        if self.verbose:
            print('Reward cum: {}'.format(numpy.sum(self.arms_reward)))

        return self.arms_visits, self.arms_reward
    
    def cumulative_mean_reward(self):
        cumsum = numpy.cumsum(self.arms_reward)
        
        x = [i for i in range(len(self.arms_visits))]
        y = cumsum / x
        z = cumsum
        
        # episode, probability, accumulated revenue
        return x, y, z
