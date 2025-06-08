import gym
import numpy as np
import pandas as pd
from gym import spaces
import random

class MultiTimeframeTradingEnv(gym.Env):
    def __init__(self, df_1h, df_4h, df_1d, df_1w):
        super().__init__()
        # 時間足データ
        self.df_1h = df_1h.sort_values("open_time").reset_index(drop=True)
        self.df_4h = df_4h.sort_values("open_time").reset_index(drop=True)
        self.df_1d = df_1d.sort_values("open_time").reset_index(drop=True)
        self.df_1w = df_1w.sort_values("open_time").reset_index(drop=True)

        self.feature_groups = {
            "price": ["open", "high", "low", "close"],
            "volume": ["volume", "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"],
            "bollinger": ["upper_band", "middle_band", "lower_band"],
            "macd": ["macd_line", "signal_line", "histogram"],
            "trend": ["adx", "plus_di", "minus_di"],
            "oscillator": ["rsi", "cci", "momentum", "mfi"],
            "volume_derived": ["obv", "cmf"],
            "stochastic": ["stoch_k", "stoch_d"],
        }

        # 4つの時間軸 * 30個の過去データ
        self.window_size = 4 * 30
        
        # action_type   [0: Long, 1: Short, 2: Spot Buy, 3: Close All, 4: Wait]
        # leverage      [used in Long/Short()]
        # amount_pct    [proportion of cash to use]
        self.action_space = spaces.MultiDiscrete([5, 10, 20])

        # ポジション: dict {type, entry_price, amount, leverage, time}
        self.positions = []  
        self.feature_names = None
        self.flat_feature_names = None

        self.fees = {"long": 0.0042, "short": 0.0042, "spot": 0.008}
        self.liquidation_threshold = 0.3

        # 観測空間の形状設定（4つの時間軸×30個の過去データ×特徴数）
        feature_count = self.df_1h.drop(columns=["open_time", "close_time", "symbol"]).shape[1] + 2
        self.observation_space = spaces.Dict({
            "obs_1h": spaces.Box(low=-np.inf, high=np.inf, shape=(30, feature_count), dtype=np.float32),
            "obs_4h": spaces.Box(low=-np.inf, high=np.inf, shape=(30, feature_count), dtype=np.float32),
            "obs_1d": spaces.Box(low=-np.inf, high=np.inf, shape=(30, feature_count), dtype=np.float32),
            "obs_1w": spaces.Box(low=-np.inf, high=np.inf, shape=(30, feature_count), dtype=np.float32),
        })

        self.reset()

    def _choose_symbol(self, df):
        if not self.symbol_cycle:
            # 使い切ったのでリセット（全シンボルをシャッフルして準備）
            self.symbol_cycle = list(df["symbol"].unique())
            random.shuffle(self.symbol_cycle)
        # リストから1つ取り出す
        return self.symbol_cycle.pop()
    
    def reset(self):
        # 開始日時を固定
        start_date = pd.to_datetime("2018-12-31")

        # 各データでstart_date以降かつ30本以上のデータがある最初のインデックスを取得
        def find_valid_start(df):
            idx = df.index[df["open_time"] >= start_date][0]
            return max(idx, 30)
        
        start_idx_1h = find_valid_start(self.df_1h)
        start_idx_4h = find_valid_start(self.df_4h)
        start_idx_1d = find_valid_start(self.df_1d)
        start_idx_1w = find_valid_start(self.df_1w)

        # すべての時間足で揃う最も遅いスタート地点に統一
        self.current_step = max(start_idx_1h, start_idx_4h, start_idx_1d, start_idx_1w)
        self.current_time = self.df_4h.loc[self.current_step, "open_time"]

        # 資産情報初期化
        self.cash = 10000
        self.asset = 0
        self.max_value = self.cash
        self.total_value = self.cash

        # symbol情報
        self.symbol_cycle = []

        return self._get_observation()

    def _extract_features(self, df, timeframe):
        chosen_symbol = self._choose_symbol(df)
        df_symbol = df[df["symbol"] == chosen_symbol]
    
        df_filtered = df_symbol[df_symbol["open_time"] <= self.current_time]
        df_window = df_filtered.tail(30).copy()

        tf_encoding = {
            "1h": 1.0,
            "4h": 4.0,
            "1d": 24.0,
            "1w": 168.0
        }
        df_window["timeframe_hours"] = tf_encoding[timeframe]

        df_window["relative_time"] = (df_window["open_time"] - self.current_time).dt.total_seconds() / 3600

        df_window = df_window.drop(columns=["open_time", "close_time", "symbol"])

        features = df_window.values.astype(np.float32)

        if self.feature_names is None:
            self.feature_names = df_window.columns.tolist()

        if len(features) < 30:
            pad_rows = 30 - len(features)
            pad = np.zeros((pad_rows, features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])

        return features
    
    def _get_observation(self):
        # 現在の時刻を更新
        self.current_time = self.df_4h.loc[self.current_step, "open_time"]

        # 各時間足から特徴抽出
        obs_1h = self._extract_features(self.df_1h, "1h")
        obs_4h = self._extract_features(self.df_4h, "4h")
        obs_1d = self._extract_features(self.df_1d, "1d")
        obs_1w = self._extract_features(self.df_1w, "1w")

        # SHAP用のフラット特徴名を初回のみ作成
        if self.flat_feature_names is None:
            self.flat_feature_names = []
            for tf in ["1h", "4h", "1d", "1w"]:
                for i in range(30):
                    for name in self.feature_names:
                        self.flat_feature_names.append(f"{tf}_{name}_t-{29 - i}")

        return {
            "obs_1h": obs_1h,
            "obs_4h": obs_4h,
            "obs_1d": obs_1d,
            "obs_1w": obs_1w,
        }
    
    def get_flat_feature_group_indices(self):
        group_indices = {}
        for group_name, features in self.feature_groups.items():
            indices = []
            for i, fname in enumerate(self.flat_feature_names):
                # 例: "1h_open_t-29" → "open"
                base_feature = fname.split('_')[1]
                if base_feature in features:
                    indices.append(i)
            group_indices[group_name] = indices
        return group_indices

    def step(self, action):
        def decode_action(action):
            action_type = action[0]
            leverage = action[1] + 1  # 1〜10に変換
            amount_pct = action[2] / 19  # 0〜1に正規化
            return action_type, leverage, amount_pct

        action_type, leverage, amount_pct = decode_action(action)

        done = False
        amount = 0
        reward = 0
        lev = leverage
        price = self.df_4h.loc[self.current_step, "close"]
        self.current_time = self.df_4h.loc[self.current_step, "open_time"]

        symbol = "BTCUSDT"  # 必要に応じて環境から取得

        if not hasattr(self, "positions"):
            self.positions = []

        if action_type in [0, 1, 2] and self.cash > 0 and amount_pct > 0:
            asset_type = ["long", "short", "spot"][action_type]
            fee = self.fees[asset_type]
            invest_amount = self.cash * np.clip(amount_pct, 0, 1)
            self.cash -= invest_amount
            actual_amount = invest_amount * (1 - fee)

            self.positions.append({
                "type": asset_type,
                "entry_price": price,
                "amount": actual_amount,
                "leverage": leverage if asset_type in ["long", "short"] else 1.0,
                "symbol": symbol,
                "time": 0
            })

            reward -= invest_amount * fee

        elif action_type == 3:  # 決済
            for pos in self.positions:
                entry = pos["entry_price"]
                amount = pos["amount"]
                lev = pos["leverage"]
                pos_type = pos["type"]

                if pos_type == "spot":
                    pnl = (price * (amount / entry)) - amount
                else:
                    diff = (price - entry) / entry
                    pnl = amount * lev * (diff if pos_type == "long" else -diff)

                reward += pnl
                if pnl < 0:
                    reward -= abs(pnl)  # 損失ペナルティ

                self.cash += amount + pnl
            self.positions = []

        elif action_type == 4:  # 待機
            for pos in self.positions:
                if pos["type"] in ["long", "short"]:
                    if self.current_time.hour % 8 == 0:
                        reward -= 0.001 * pos["amount"]
                elif pos["type"] == "spot":
                    reward += 0.00004 * (price * (pos["amount"] / pos["entry_price"]))
                pos["time"] += 1

        # ロスカット処理
        survivors = []
        for pos in self.positions:
            entry = pos["entry_price"]
            pos_type = pos["type"]
            drawdown = -((price - entry) / entry) if pos_type == "long" else (price - entry) / entry

            if drawdown > self.liquidation_threshold:
                reward -= 3 * amount
                self.cash += amount * (1 - drawdown)
                continue

            # 絶対額ベースのロスカット（評価額が70%未満になったら）
            if pos_type == "spot":
                current_value = price * (amount / entry)
            else:
                diff = (price - entry) / entry
                pnl = amount * lev * (diff if pos_type == "long" else -diff)
                current_value = amount + pnl

            if current_value < 0.7 * amount:
                reward -= 3 * amount
                self.cash += max(0, current_value)
                continue
            survivors.append(pos)
        self.positions = survivors

        # 資産更新
        total_value = self.cash
        for pos in self.positions:
            entry = pos["entry_price"]
            pos_type = pos["type"]
            amount = pos["amount"]
            lev = pos["leverage"]

            if pos_type == "spot":
                total_value += price * (amount / entry)
            else:
                diff = (price - entry) / entry
                pnl = amount * lev * (diff if pos_type == "long" else -diff)
                total_value += amount + pnl
        self.total_value = total_value

        # 最高資産更新ボーナス
        if self.total_value > self.max_value:
            reward += 0.1 * self.total_value
            self.max_value = self.total_value

        # ベンチマーク報酬
        if self.current_step > 1:
            prev_price = self.df_4h.loc[self.current_step - 1, "close"]
            benchmark_return = price / prev_price
            if benchmark_return > 1:
                reward += 0.01 * (benchmark_return - 1) * self.total_value

        # 各資産クラスごとの含み損益率を記録
        self.asset_returns = {}
        for pos in self.positions:
            symbol = pos["symbol"]
            entry = pos["entry_price"]
            pos_type = pos["type"]

            if pos_type == "spot":
                ret = (price / entry) - 1
            else:
                diff = (price - entry) / entry
                ret = diff if pos_type == "long" else -diff

            if symbol not in self.asset_returns:
                self.asset_returns[symbol] = []
            self.asset_returns[symbol].append(ret)

        self.current_step += 1
        done = self.current_step >= len(self.df_4h) - 1
        obs = self._get_observation()

        if self.cash < 10:
            reward -= 1000  # 大きめの罰則
            done = True

        return obs, reward, done, {
            "cash": self.cash,
            "positions": self.positions,
            "value": self.total_value,
            "step": self.current_step,
            "asset_returns": self.asset_returns
        }
