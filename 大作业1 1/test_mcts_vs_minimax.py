#!/usr/bin/env python3
"""
测试 MCTS 智能体与 Minimax 智能体的对战
"""

from dlgo import GameState, Player, Point
from agents.mcts_agent import MCTSAgent
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
    print("开始测试 MCTS 智能体 vs Minimax 智能体...")
    
    # 创建游戏状态
    game = GameState.new_game(5)
    
    # 创建智能体
    mcts_agent = MCTSAgent(num_rounds=100)
    minimax_agent = MinimaxAgent(max_depth=3, use_cache=False)
    
    # 对弈循环
    move_count = 0
    max_moves = 50  # 防止无限循环
    
    while not game.is_over() and move_count < max_moves:
        print(f"\n=== Move {move_count + 1}, {game.next_player.name} ===")
        print_board(game)
        
        if game.next_player == Player.black:
            # MCTS 智能体执黑
            move = mcts_agent.select_move(game)
            print(f"MCTS 选择: {move}")
        else:
            # Minimax 智能体执白
            move = minimax_agent.select_move(game)
            print(f"Minimax 选择: {move}")
        
        game = game.apply_move(move)
        move_count += 1
    
    # 打印终局
    print("\n=== 终局 ===")
    print_board(game)
    winner = game.winner()
    if winner:
        print(f"胜者: {winner.name}")
    else:
        print("平局")


if __name__ == "__main__":
    main()
