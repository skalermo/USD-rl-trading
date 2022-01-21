class RandomAgent:
    def __init__(self, env, **kwargs):
        self.env = env

    def learn(self, **kwargs):
        pass

    def predict(self, _state):
        return self.env.action_space.sample(), None
