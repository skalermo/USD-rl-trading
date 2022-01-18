from gym_anytrading.envs.stocks_env import StocksEnv

from src.StockTradingGraph import StockTradingGraph

# todo adjust script to be compatible with gym_anytrading's StockEnv
LOOKBACK_WINDOW_SIZE = 40
INITIAL_ACCOUNT_BALANCE = 10000


class StocksEnvCustom(StocksEnv):
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.visualization = None
        self.current_step = 0
        self.trades = []

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        # if mode == 'file':
        #     self._render_to_file(kwargs.get('filename', 'render.txt'))

        if mode == 'live':
            if self.visualization is None:
                self.visualization = StockTradingGraph(
                    self.df, kwargs.get('title', None))

            if self.current_step > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.current_step, self.net_worth, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None
