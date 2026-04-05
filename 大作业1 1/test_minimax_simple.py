#!/usr/bin/env python3
"""
测试 Minimax 智能体的简单脚本
"""

from dlgo import GameState, Player, Point
from agents.minimax_agent import MinimaxAgent


def print_board(game_state):
    """打印棋盘"""
    board = game_state.board
    print("  ", end="")
    for c in range(1, board.num_cols + 1):
        print(f"{c:2}", end="")
    print()

    for r in range(1, board.num_rows + 1):
        print(f"{r:2}", end="")
        for c in range(1, board.num_cols + 1):
            stone = board.get(Point(r, c))
            if stone == Player.black:
                print(" X", end="")
            elif stone == Player.white:
                print(" O", end="")
            else:
                print(" .", end="")
        print()


def main():
    """主函数"""
    print("开始测试 Minimax 智能体 (深度 1)...")
    
    # 创建游戏状态
    game = GameState.new_game(5)
    
    # 创建智能体（使用较小的深度）
    minimax_agent = MinimaxAgent(max_depth=1, use_cache=False)
    
    # 测试第一步
    print("\n=== 初始局面 ===")
    print_board(game)
    
    move = minimax_agent.select_move(game)
    print(f"\nMinimax 选择: {move}")
    
    # 应用棋步
    game = game.apply_move(move)
    print("\n=== 应用棋步后 ===")
    print_board(game)


if __name__ == "__main__":
    main()
