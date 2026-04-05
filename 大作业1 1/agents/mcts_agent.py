"""
MCTS (蒙特卡洛树搜索) 智能体模板。

学生作业：完成 MCTS 算法的核心实现。
参考：《深度学习与围棋》第 4 章
"""

import math
import random
from dlgo.gotypes import Player, Point
from dlgo.goboard import GameState, Move

__all__ = ["MCTSAgent"]



class MCTSNode:
    """
    MCTS 树节点。


    属性：
        game_state: 当前局面
        parent: 父节点（None 表示根节点）
        children: 子节点列表
        visit_count: 访问次数
        value_sum: 累积价值（胜场数）
        prior: 先验概率（来自策略网络，可选）
        move_score: 棋步评分
        board_score: 棋盘评分
    """

    def __init__(self, game_state, parent=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.move_score = 0
        self.board_score = 0
        # 优先尝试非 pass、非 resign 的棋步
        legal_moves = game_state.legal_moves()
        playable_moves = [move for move in legal_moves if move.is_play]
        non_playable_moves = [move for move in legal_moves if not move.is_play]
        self._untried_moves = playable_moves + non_playable_moves

    @property
    def value(self):
        """计算平均价值 = value_sum / visit_count，防止除零。"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否为叶节点（未展开）。"""
        return len(self.children) == 0

    def is_terminal(self):
        """是否为终局节点。"""
        return self.game_state.is_over()

    def best_child(self, c=1.414):
        """
        选择最佳子节点（UCT 算法）。

        UCT = value + c * sqrt(ln(parent_visits) / visits)

        Args:
            c: 探索常数（默认 sqrt(2)）

        Returns:
            最佳子节点
        """
        # 优先选择未被访问过的子节点
        unvisited_children = [child for child in self.children if child.visit_count == 0]
        if unvisited_children:
            return random.choice(unvisited_children)
        
        # 计算UCT值
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if self.visit_count > 0 and child.visit_count > 0:
                uct_score = child.value + c * math.sqrt(math.log(self.visit_count) / child.visit_count)
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
        
        # 如果所有子节点都被访问过，返回最佳的那个
        if best_child:
            return best_child
        
        # 如果没有子节点，返回None
        return None

    def expand(self):
        """
        展开节点：为所有合法棋步创建子节点。

        Returns:
            新创建的子节点（用于后续模拟）
        """
        if not self._untried_moves:
            return None
        move = self._untried_moves.pop()
        next_state = self.game_state.apply_move(move)
        child_node = MCTSNode(next_state, parent=self)
        # 计算棋步评分和棋盘评分
        child_node.move_score = self._evaluate_move(self.game_state, next_state, self.game_state.next_player)
        child_node.board_score = self._evaluate_board(next_state, self.game_state.next_player)
        self.children.append(child_node)
        return child_node

    def backup(self, value):
        """
        反向传播：更新从当前节点到根节点的统计。

        Args:
            value: 从当前局面模拟得到的结果（1=胜，0=负，0.5=和）
        """
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # 对父节点来说，价值是相反的
            self.parent.backup(1 - value)

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

    def _evaluate_board(self, game_state, player):
        """
        评估棋盘的质量。

        Args:
            game_state: 游戏状态
            player: 玩家

        Returns:
            棋盘评分
        """
        board = game_state.board
        score = 0
        
        # 计算棋子数量
        player_stones = 0
        opponent_stones = 0
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                color = board.get(Point(r, c))
                if color == player:
                    player_stones += 1
                elif color == player.other:
                    opponent_stones += 1
        score += (player_stones - opponent_stones) * 2
        
        # 计算控制区域
        for r in range(1, board.num_rows + 1):
            for c in range(1, board.num_cols + 1):
                color = board.get(Point(r, c))
                if color is None:
                    # 空点评估：根据周围棋子的颜色
                    neighbors = Point(r, c).neighbors()
                    player_neighbors = 0
                    opponent_neighbors = 0
                    for neighbor in neighbors:
                        if board.is_on_grid(neighbor):
                            neighbor_color = board.get(neighbor)
                            if neighbor_color == player:
                                player_neighbors += 1
                            elif neighbor_color == player.other:
                                opponent_neighbors += 1
                    if player_neighbors > opponent_neighbors:
                        score += 1
                    elif opponent_neighbors > player_neighbors:
                        score -= 1
        
        return score


class MCTSAgent:
    """
    MCTS 智能体。

    属性：
        num_rounds: 每次决策的模拟轮数
        temperature: 温度参数（控制探索程度）
    """

    def __init__(self, num_rounds=5000, temperature=1.0):
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state: GameState) -> Move:
        """
        为当前局面选择最佳棋步。

        流程：
            1. 创建根节点
            2. 预展开所有可能的棋步
            3. 进行 num_rounds 轮模拟：
               a. Selection: 用 UCT 选择路径到叶节点
               b. Expansion: 展开叶节点
               c. Simulation: 随机模拟至终局
               d. Backup: 反向传播结果
            4. 选择访问次数最多的棋步

        Args:
            game_state: 当前游戏状态

        Returns:
            选定的棋步
        """
        root = MCTSNode(game_state)

        # 预展开所有可能的棋步
        while root._untried_moves:
            root.expand()

        # 进行模拟
        for _ in range(self.num_rounds):
            node = root
            # 选择阶段：沿着UCT值最大的路径前进
            while not node.is_leaf() and not node.is_terminal():
                best_child = node.best_child()
                if best_child is None:
                    break
                node = best_child
            
            # 扩展阶段：如果不是终局，展开节点
            if not node.is_terminal():
                expanded_node = node.expand()
                if expanded_node is not None:
                    node = expanded_node
            
            # 模拟阶段：随机走子至终局
            value = self._simulate(node.game_state)
            
            # 反向传播阶段：更新统计信息
            node.backup(value)

        # 选择最佳棋步
        return self._select_best_move(root)

    def _simulate(self, game_state):
        """
        快速模拟（Rollout）：随机走子至终局。

        【第二小问要求】
        标准 MCTS 使用完全随机走子，但需要实现至少两种优化方法：
        1. 启发式走子策略（如：优先选有气、不自杀、提子的走法）
        2. 限制模拟深度（如：最多走 20-30 步后停止评估）
        3. 其他：快速走子评估（RAVE）、池势启发等

        Args:
            game_state: 起始局面

        Returns:
            从当前玩家视角的结果（1=胜, 0=负, 0.5=和）
        """
        current_state = game_state
        max_depth = 30
        depth = 0

        while not current_state.is_over() and depth < max_depth:
            moves = current_state.legal_moves()
            if not moves:
                break
            
            # 启发式走子策略
            # 优先选择非pass、非resign的走法
            playable_moves = [move for move in moves if move.is_play]
            if playable_moves:
                # 70% 的概率选择启发式最佳走法，30% 的概率随机选择
                if random.random() < 0.7:
                    # 评估每个走法
                    move_scores = []
                    for move in playable_moves:
                        next_state = current_state.apply_move(move)
                        score = self._evaluate_move(current_state, next_state, current_state.next_player)
                        move_scores.append((score, move))
                    # 选择评分最高的走法
                    move_scores.sort(reverse=True)
                    move = move_scores[0][1]
                else:
                    # 随机选择一个可落子的位置
                    move = random.choice(playable_moves)
            else:
                # 如果没有可落子的位置，选择pass
                move = Move.pass_turn()
            
            current_state = current_state.apply_move(move)
            depth += 1

        # 如果达到最大深度，根据当前局面评估
        if not current_state.is_over():
            # 改进的评估：不仅考虑棋子数量，还考虑控制的区域
            board = current_state.board
            black_score = 0
            white_score = 0
            
            for r in range(1, board.num_rows + 1):
                for c in range(1, board.num_cols + 1):
                    color = board.get(Point(r, c))
                    if color == Player.black:
                        black_score += 2  # 棋子本身价值
                    elif color == Player.white:
                        white_score += 2  # 棋子本身价值
                    else:
                        # 空点评估：根据周围棋子的颜色
                        neighbors = Point(r, c).neighbors()
                        black_neighbors = 0
                        white_neighbors = 0
                        for neighbor in neighbors:
                            if board.is_on_grid(neighbor):
                                neighbor_color = board.get(neighbor)
                                if neighbor_color == Player.black:
                                    black_neighbors += 1
                                elif neighbor_color == Player.white:
                                    white_neighbors += 1
                        if black_neighbors > white_neighbors:
                            black_score += 1
                        elif white_neighbors > black_neighbors:
                            white_score += 1
            
            if game_state.next_player == Player.black:
                return 1.0 if black_score > white_score else 0.0 if white_score > black_score else 0.5
            else:
                return 1.0 if white_score > black_score else 0.0 if black_score > white_score else 0.5
        
        # 终局结果
        winner = current_state.winner()
        if winner == game_state.next_player:
            return 1.0
        elif winner is None:
            return 0.5
        else:
            return 0.0

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

    def _select_best_move(self, root):
        """
        根据访问次数选择最佳棋步。

        Args:
            root: MCTS 树根节点

        Returns:
            最佳棋步
        """
        if not root.children:
            return Move.pass_turn()
        
        # 收集所有子节点及其对应的棋步
        child_move_pairs = []
        for move in root.game_state.legal_moves():
            next_state = root.game_state.apply_move(move)
            for child in root.children:
                if child.game_state.board.zobrist_hash() == next_state.board.zobrist_hash():
                    child_move_pairs.append((child, move))
                    break
        
        if not child_move_pairs:
            return Move.pass_turn()
        
        # 优先选择非 pass、非 resign 的棋步
        playable_pairs = [(child, move) for (child, move) in child_move_pairs if move.is_play]
        if playable_pairs:
            best_child, best_move = max(playable_pairs, key=lambda x: x[0].visit_count)
            return best_move
        
        # 如果没有可落子的棋步，选择 pass
        best_child, best_move = max(child_move_pairs, key=lambda x: x[0].visit_count)
        return best_move
