"""
第三小问（选做）：Minimax 智能体

实现 Minimax + Alpha-Beta 剪枝算法，与 MCTS 对比效果。
可选实现，用于对比不同搜索算法的差异。

参考：《深度学习与围棋》第 3 章
"""

import random
from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MinimaxAgent", "GameResultCache"]


class GameResultCache:
    """
    置换表（Transposition Table）：缓存已计算的局面结果。
    
    用于加速搜索，避免重复计算相同局面。
    
    属性：
        cache: 字典 {zobrist_hash: (depth, value, flag)}
        hits: 缓存命中次数
        misses: 缓存未命中次数
        max_size: 缓存最大大小
    """
    
    def __init__(self, max_size=1000000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def put(self, game_state, depth, value, flag='exact'):
        """
        将局面结果存入缓存。
        
        Args:
            game_state: 游戏状态
            depth: 搜索深度
            value: 评估值
            flag: 'exact'/'lower'/'upper'（精确值/下界/上界）
        """
        if len(self.cache) >= self.max_size:
            # 缓存满时清空（简单策略）
            self.cache.clear()
        
        key = game_state.board.zobrist_hash()
        # 只在深度足够时才缓存
        if key not in self.cache or depth >= self.cache[key][0]:
            self.cache[key] = (depth, value, flag)
    
    def get(self, game_state, depth):
        """
        从缓存获取局面结果。
        
        Args:
            game_state: 游戏状态
            depth: 搜索深度
        
        Returns:
            (value, found) - 评估值和是否找到
        """
        key = game_state.board.zobrist_hash()
        
        if key in self.cache:
            cached_depth, cached_value, flag = self.cache[key]
            # 只在深度足够时才信任缓存
            if cached_depth >= depth:
                self.hits += 1
                return cached_value, True
        
        self.misses += 1
        return None, False
    
    def clear(self):
        """清空缓存。"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self):
        """返回缓存统计信息。"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


class MinimaxAgent:
    """
    Minimax 智能体（带 Alpha-Beta 剪枝 + 置换表缓存）。

    属性：
        max_depth: 搜索最大深度
        evaluator: 局面评估函数
        cache: 置换表（缓存已计算的局面）
    """

    def __init__(self, max_depth=3, evaluator=None, use_cache=True):
        self.max_depth = max_depth
        self.evaluator = evaluator or self._default_evaluator
        # 默认使用缓存
        self.cache = GameResultCache() if use_cache else None

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return None
        
        # 过滤掉 resign 棋步，除非没有其他选择
        play_moves = [move for move in legal_moves if move.is_play]
        if not play_moves:
            return legal_moves[0]
        
        best_move = None
        best_value = -float('inf')
        
        # 对棋步进行排序，优化剪枝效率
        ordered_moves = self._get_ordered_moves(game_state)
        # 确保只考虑 play 类型的棋步
        ordered_play_moves = [move for move in ordered_moves if move.is_play]
        if not ordered_play_moves:
            return legal_moves[0]
        
        # 当前玩家是 maximizing player
        for move in ordered_play_moves:
            next_state = game_state.apply_move(move)
            value = self.alphabeta(next_state, self.max_depth - 1, -float('inf'), float('inf'), False)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else ordered_play_moves[0]

    def minimax(self, game_state, depth, maximizing_player):
        """
        基础 Minimax 算法：搜索所有可能的棋步组合。
        
        【算法原理】
        Minimax 基于博弈论中的"零和游戏"假设：
        - 我方(MAX)试图最大化局面评分
        - 对方(MIN)试图最小化局面评分
        - 双方都按最优策略走子
        
        递归结构：
        ```
        Minimax(state, depth, MAX):
            if depth == 0 or terminal:
                return evaluate(state)
            best = -∞
            for each move:
                best = max(best, Minimax(next_state, depth-1, MIN))
            return best
        
        Minimax(state, depth, MIN):
            if depth == 0 or terminal:
                return evaluate(state)
            best = +∞
            for each move:
                best = min(best, Minimax(next_state, depth-1, MAX))
            return best
        ```

        时间复杂度：O(b^d)，其中 b=分支因子，d=深度
        对于围棋 5x5 盘，b≈100-200（分支数多）

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度（>0：继续搜索，=0：叶节点）
            maximizing_player: True=最大化方(我方)，False=最小化方(对手)

        Returns:
            该局面的评估值（正数对我方有利，负数对我方不利）
        """
        # 【终止条件】
        if depth == 0 or game_state.is_over():
            return self.evaluator(game_state)
        
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return self.evaluator(game_state)
        
        if maximizing_player:
            # 【MAX节点】寻求最大值
            max_value = -float('inf')
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = self.minimax(next_state, depth - 1, False)  # 递归到MIN节点
                max_value = max(max_value, value)
            return max_value
        else:
            # 【MIN节点】寻求最小值
            min_value = float('inf')
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = self.minimax(next_state, depth - 1, True)  # 递归到MAX节点
                min_value = min(min_value, value)
            return min_value

    def alphabeta(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Alpha-Beta 剪枝优化版 Minimax（带置换表缓存）。

        【核心优化】
        1. Alpha-Beta 剪枝：β剪枝（最大化方）和α剪枝（最小化方）
        2. 置换表缓存：避免重复计算相同局面
        
        Alpha-Beta 剪枝原理：
        - Alpha：最大化方能保证的最小值（下界）
        - Beta：最小化方能保证的最大值（上界）
        - 当 value >= beta（最大化方）或 value <= alpha（最小化方）时，可剪枝

        Args:
            game_state: 当前局面
            depth: 剩余搜索深度
            alpha: 当前最大下界
            beta: 当前最小上界
            maximizing_player: 是否在当前层最大化

        Returns:
            该局面的评估值
        """
        # 【优化1】检查缓存
        if self.cache:
            cached_value, found = self.cache.get(game_state, depth)
            if found:
                return cached_value
        
        # 终局条件：达到最大深度或游戏结束
        if depth == 0 or game_state.is_over():
            value = self.evaluator(game_state)
            if self.cache:
                self.cache.put(game_state, depth, value, 'exact')
            return value
        
        legal_moves = self._get_ordered_moves(game_state)
        if not legal_moves:
            value = self.evaluator(game_state)
            if self.cache:
                self.cache.put(game_state, depth, value, 'exact')
            return value
        
        if maximizing_player:
            # 【最大化方】寻求最大值
            value = -float('inf')
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = max(value, self.alphabeta(next_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                # 【β剪枝】：如果 value >= beta，表示对手不会选这条路
                if value >= beta:
                    break
            
            if self.cache:
                self.cache.put(game_state, depth, value, 'exact')
            return value
        else:
            # 【最小化方】寻求最小值
            value = float('inf')
            for move in legal_moves:
                next_state = game_state.apply_move(move)
                value = min(value, self.alphabeta(next_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                # 【α剪枝】：如果 value <= alpha，表示我方不会选这条路
                if value <= alpha:
                    break
            
            if self.cache:
                self.cache.put(game_state, depth, value, 'exact')
            return value

    def _default_evaluator(self, game_state):
        """
        默认局面评估函数：将非终局局面转换为评分。
        
        【评估策略】
        1. 终局评分（绝对）：
           - 我方赢：+1000
           - 我方输：-1000
           - 平局：0
        
        2. 非终局评分（相对）：
           评分 = 石子数差×5 + 气数差×1 + 边缘位置 bonus + Euler 数评估
           
           权重解释：
           - 石子数（×5）：占地的直接指标
           - 气数（×1）：棋串生存力的指标
           - 边缘位置 bonus：占据有利位置的奖励
           - Euler 数：评估领地控制

        Args:
            game_state: 游戏状态

        Returns:
            评估值（浮点数，>0=我方有利，<0=我方不利）
        """
        # 【情况1】游戏已结束
        if game_state.is_over():
            winner = game_state.winner()
            if winner == game_state.next_player:
                return 1000  # 我们赢了
            elif winner == game_state.next_player.other:
                return -1000  # 对手赢了
            else:
                return 0  # 平局
        
        board = game_state.board
        current_player = game_state.next_player
        opponent = current_player.other
        
        # 【情况2】非终局，统计棋子和气数
        our_stones = 0
        opponent_stones = 0
        our_liberties = 0
        opponent_liberties = 0
        our_edge = 0
        opponent_edge = 0
        
        evaluated_strings = set()  # 避免重复计算同一棋串
        
        for row in range(1, board.num_rows + 1):
            for col in range(1, board.num_cols + 1):
                point = Point(row, col)
                stone = board.get(point)
                
                if stone == current_player:
                    our_stones += 1
                    # 边缘位置 bonus
                    if row == 1 or row == board.num_rows or col == 1 or col == board.num_cols:
                        our_edge += 1
                    try:
                        go_string = board.get_go_string(point)
                        # 避免重复计算同一棋串的气
                        if go_string and id(go_string) not in evaluated_strings:
                            evaluated_strings.add(id(go_string))
                            our_liberties += len(go_string.liberties)
                    except:
                        pass
                        
                elif stone == opponent:
                    opponent_stones += 1
                    # 边缘位置 bonus
                    if row == 1 or row == board.num_rows or col == 1 or col == board.num_cols:
                        opponent_edge += 1
                    try:
                        go_string = board.get_go_string(point)
                        if go_string and id(go_string) not in evaluated_strings:
                            evaluated_strings.add(id(go_string))
                            opponent_liberties += len(go_string.liberties)
                    except:
                        pass
        
        # 计算 Euler 数
        euler_score = self._calculate_euler_number(game_state, current_player)
        
        # 【评分公式】
        # 权重：石子数×5（更重要），气数×1（参考），边缘位置 bonus，Euler 数
        score = (our_stones - opponent_stones) * 5 + \
                (our_liberties - opponent_liberties) * 1 + \
                (our_edge - opponent_edge) * -1 + \
                euler_score * -4
        
        return score

    def _calculate_euler_number(self, game_state, player):
        """
        计算 Euler 数，用于评估领地控制。
        
        Args:
            game_state: 游戏状态
            player: 当前玩家
            
        Returns:
            Euler 数评分
        """
        board = game_state.board
        size = board.num_rows
        
        # 创建扩展棋盘，方便边界处理
        extended_board = [[0] * (size + 2) for _ in range(size + 2)]
        for row in range(1, size + 1):
            for col in range(1, size + 1):
                point = Point(row, col)
                stone = board.get(point)
                if stone == player:
                    extended_board[row][col] = 1
                elif stone == player.other:
                    extended_board[row][col] = 2
        
        q1_p1 = 0
        q3_p1 = 0
        qd_p1 = 0
        q1_p2 = 0
        q3_p2 = 0
        qd_p2 = 0
        
        # 遍历所有 2x2 窗口
        for i in range(1, size + 1):
            for j in range(1, size + 1):
                # 获取 2x2 窗口
                state = [
                    [extended_board[i][j], extended_board[i][j+1]],
                    [extended_board[i+1][j], extended_board[i+1][j+1]]
                ]
                
                # 计算 Q1, Q3, Qd
                q1_p1 += self._q1(state, 1)
                q3_p1 += self._q3(state, 1)
                qd_p1 += self._qd(state, 1)
                
                q1_p2 += self._q1(state, 2)
                q3_p2 += self._q3(state, 2)
                qd_p2 += self._qd(state, 2)
        
        # 计算 Euler 数
        euler_num = (2 * 1 * q3_p1 - (q1_p2 - qd_p2 + 2 * q3_p2) + q1_p1 - qd_p1) / 4
        return euler_num

    def _q1(self, state, player):
        """
        计算 Q1 值：单个棋子的情况。
        
        Args:
            state: 2x2 窗口
            player: 玩家（1 或 2）
            
        Returns:
            1 如果满足条件，否则 0
        """
        bl, br, tl, tr = state[0][0], state[0][1], state[1][0], state[1][1]
        count = 0
        if bl == player and br != player and tl != player and tr != player:
            count = 1
        if bl != player and br == player and tl != player and tr != player:
            count = 1
        if bl != player and br != player and tl == player and tr != player:
            count = 1
        if bl != player and br != player and tl != player and tr == player:
            count = 1
        return count

    def _q3(self, state, player):
        """
        计算 Q3 值：对角棋子的情况。
        
        Args:
            state: 2x2 窗口
            player: 玩家（1 或 2）
            
        Returns:
            1 如果满足条件，否则 0
        """
        bl, br, tl, tr = state[0][0], state[0][1], state[1][0], state[1][1]
        count = 0
        if bl == player and br != player and tl != player and tr == player:
            count = 1
        if bl != player and br == player and tl == player and tr != player:
            count = 1
        return count

    def _qd(self, state, player):
        """
        计算 Qd 值：三个棋子的情况。
        
        Args:
            state: 2x2 窗口
            player: 玩家（1 或 2）
            
        Returns:
            1 如果满足条件，否则 0
        """
        bl, br, tl, tr = state[0][0], state[0][1], state[1][0], state[1][1]
        count = 0
        if bl == player and br == player and tl == player and tr != player:
            count = 1
        if bl != player and br == player and tl == player and tr == player:
            count = 1
        if bl == player and br != player and tl == player and tr == player:
            count = 1
        if bl == player and br == player and tl != player and tr == player:
            count = 1
        return count

    def _get_ordered_moves(self, game_state):
        """
        获取排序后的候选棋步（用于优化剪枝效率）。

        好的排序能让 Alpha-Beta 剪掉更多分支。

        Args:
            game_state: 游戏状态

        Returns:
            按启发式排序的棋步列表
        """
        legal_moves = game_state.legal_moves()
        
        # 评估每个棋步
        move_scores = []
        for move in legal_moves:
            if move.is_play:
                # 评估提子情况
                next_state = game_state.apply_move(move)
                score = self._evaluate_move(game_state, next_state, game_state.next_player)
                move_scores.append((score, move))
            else:
                # pass 和 resign 评分较低
                move_scores.append((-100, move))
        
        # 按评分降序排序
        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        # 返回排序后的棋步
        return [move for score, move in move_scores]

    def _evaluate_move(self, current_state, next_state, player):
        """
        评估棋步的质量。

        Args:
            current_state: 当前游戏状态
            next_state: 下一步游戏状态
            player: 落子玩家

        Returns:
            棋步评分
        """
        score = 0
        
        # 检查是否提子
        current_board = current_state.board
        next_board = next_state.board
        
        # 计算提子数
        current_stones = 0
        next_stones = 0
        for r in range(1, current_board.num_rows + 1):
            for c in range(1, current_board.num_cols + 1):
                if current_board.get(Point(r, c)) == player.other:
                    current_stones += 1
                if next_board.get(Point(r, c)) == player.other:
                    next_stones += 1
        capture_count = current_stones - next_stones
        if capture_count > 0:
            score += capture_count * 10
        
        # 检查是否占据角落或边缘
        move_point = None
        for move in current_state.legal_moves():
            if move.is_play:
                test_state = current_state.apply_move(move)
                if test_state.board.zobrist_hash() == next_state.board.zobrist_hash():
                    move_point = move.point
                    break
        
        if move_point:
            # 角落
            if (move_point.row == 1 or move_point.row == current_board.num_rows) and \
               (move_point.col == 1 or move_point.col == current_board.num_cols):
                score += 5
            # 边缘
            elif (move_point.row == 1 or move_point.row == current_board.num_rows) or \
                 (move_point.col == 1 or move_point.col == current_board.num_cols):
                score += 3
        
        return score
