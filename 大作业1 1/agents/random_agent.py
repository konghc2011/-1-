"""
第一小问（必做）：随机 AI

基于随机落子（但需满足规则）的基础围棋 AI
用于验证规则调用和基础设施正常工作。
"""

import random
from dlgo import GameState, Move

__all__ = ["RandomAgent"]


class RandomAgent:
    """
    随机落子智能体 - 第一小问实现

    从所有合法棋步中均匀随机选择，包括：
    - 正常落子
    - 停一手 (pass)
    - 认输 (resign)
    """

    def __init__(self):
        """初始化随机智能体（无需特殊参数）。"""
        self.name = "RandomAgent"

    def select_move(self, game_state: GameState) -> Move:
        """
        从当前局面的所有合法棋步中随机选择一个。

        Args:
            game_state: 当前游戏状态

        Returns:
            随机选择的合法 Move
        """
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            raise ValueError("当前游戏状态没有合法棋步")
        return random.choice(legal_moves)


# 便捷函数（向后兼容 play.py）
def random_agent(game_state: GameState) -> Move:
    """函数接口，兼容 play.py 的调用方式"""
    agent = RandomAgent()
    return agent.select_move(game_state)