#!/usr/bin/env python3
"""
围棋人机对战可视化工具

使用 Tkinter 实现图形化界面，支持与不同类型的 AI 智能体对战：
- 随机智能体 (RandomAgent)
- MCTS 智能体 (MCTSAgent)
- Minimax 智能体 (MinimaxAgent)

功能：
- 显示棋盘和棋子
- 支持鼠标点击落子
- 显示当前回合、提子数等信息
- 支持新游戏、悔棋等功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dlgo.goboard import GameState, Move
from dlgo.gotypes import Player, Point
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent


class GoVisualizer:
    """围棋可视化界面"""
    
    def __init__(self, root):
        """初始化可视化界面"""
        self.root = root
        self.root.title("围棋人机对战")
        self.root.geometry("600x700")
        
        # 游戏设置
        self.board_size = 5
        self.current_player = Player.black
        self.game_state = None
        self.ai_agent = None
        self.ai_type = "mcts"
        self.game_history = []
        
        # 颜色设置
        self.board_color = "#D4A76A"  # 棋盘颜色
        self.stone_colors = {
            Player.black: "#000000",  # 黑棋
            Player.white: "#FFFFFF"   # 白棋
        }
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        self.control_frame = ttk.Frame(self.main_frame, padding="10")
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # 智能体选择
        ttk.Label(self.control_frame, text="AI 类型:").grid(row=0, column=0, padx=5, pady=5)
        self.ai_type_var = tk.StringVar(value="mcts")
        ai_types = ["random", "mcts", "minimax"]
        self.ai_type_combo = ttk.Combobox(
            self.control_frame, 
            textvariable=self.ai_type_var, 
            values=ai_types, 
            state="readonly"
        )
        self.ai_type_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 新游戏按钮
        self.new_game_btn = ttk.Button(
            self.control_frame, 
            text="新游戏", 
            command=self.new_game
        )
        self.new_game_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 悔棋按钮
        self.undo_btn = ttk.Button(
            self.control_frame, 
            text="悔棋", 
            command=self.undo_move, 
            state=tk.DISABLED
        )
        self.undo_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 游戏信息
        self.info_frame = ttk.Frame(self.main_frame, padding="10")
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="准备开始新游戏")
        self.status_label = ttk.Label(self.info_frame, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack(anchor=tk.W)
        
        # 棋盘画布
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bg=self.board_color, 
            highlightthickness=2, 
            highlightbackground="#8B4513"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_board_click)
        
        # 初始化棋盘
        self.init_board()
        
        # 开始新游戏
        self.new_game()
    
    def init_board(self):
        """初始化棋盘"""
        self.canvas.delete("all")
        
        # 计算棋盘尺寸和格子大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 确保画布大小合适
        if canvas_width < 400 or canvas_height < 400:
            self.root.after(100, self.init_board)
            return
        
        # 计算格子大小
        self.cell_size = min(
            (canvas_width - 40) / (self.board_size + 1),
            (canvas_height - 40) / (self.board_size + 1)
        )
        
        # 计算棋盘起始位置
        self.board_x = (canvas_width - (self.board_size - 1) * self.cell_size) / 2
        self.board_y = (canvas_height - (self.board_size - 1) * self.cell_size) / 2
        
        # 绘制棋盘线条
        for i in range(self.board_size):
            # 横线
            y = self.board_y + i * self.cell_size
            self.canvas.create_line(
                self.board_x, y, 
                self.board_x + (self.board_size - 1) * self.cell_size, y,
                width=2
            )
            # 竖线
            x = self.board_x + i * self.cell_size
            self.canvas.create_line(
                x, self.board_y, 
                x, self.board_y + (self.board_size - 1) * self.cell_size,
                width=2
            )
        
        # 绘制星位（如果是 5x5 棋盘，星位在 (2,2), (2,4), (4,2), (4,4)）
        if self.board_size >= 5:
            star_points = [(2, 2), (2, 4), (4, 2), (4, 4)]
            for r, c in star_points:
                x = self.board_x + (c - 1) * self.cell_size
                y = self.board_y + (r - 1) * self.cell_size
                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4, 
                    fill="#000000"
                )
    
    def new_game(self):
        """开始新游戏"""
        # 创建新的游戏状态
        self.game_state = GameState.new_game(self.board_size)
        self.game_history = []
        self.current_player = Player.black
        
        # 选择 AI 类型
        self.ai_type = self.ai_type_var.get()
        
        # 初始化 AI 智能体
        if self.ai_type == "random":
            self.ai_agent = RandomAgent()
        elif self.ai_type == "mcts":
            self.ai_agent = MCTSAgent(num_rounds=1000)
        elif self.ai_type == "minimax":
            self.ai_agent = MinimaxAgent(max_depth=2)
        
        # 更新状态信息
        self.update_status()
        
        # 绘制棋盘
        self.draw_board()
        
        # 启用悔棋按钮
        self.undo_btn.config(state=tk.NORMAL)
    
    def draw_board(self):
        """绘制棋盘和棋子"""
        self.init_board()
        
        if not self.game_state:
            return
        
        board = self.game_state.board
        
        # 绘制棋子
        for r in range(1, self.board_size + 1):
            for c in range(1, self.board_size + 1):
                point = Point(r, c)
                stone = board.get(point)
                if stone:
                    x = self.board_x + (c - 1) * self.cell_size
                    y = self.board_y + (r - 1) * self.cell_size
                    radius = self.cell_size * 0.4
                    
                    # 绘制棋子
                    self.canvas.create_oval(
                        x - radius, y - radius, 
                        x + radius, y + radius, 
                        fill=self.stone_colors[stone],
                        outline="#000000",
                        width=1
                    )
    
    def on_board_click(self, event):
        """处理鼠标点击事件"""
        if not self.game_state or self.game_state.is_over():
            return
        
        # 计算点击位置对应的棋盘坐标
        x = event.x - self.board_x
        y = event.y - self.board_y
        
        col = int(round(x / self.cell_size)) + 1
        row = int(round(y / self.cell_size)) + 1
        
        # 检查坐标是否有效
        if not (1 <= row <= self.board_size and 1 <= col <= self.board_size):
            return
        
        # 创建落子动作
        move = Move.play(Point(row, col))
        
        # 检查是否为合法棋步
        if not self.game_state.is_valid_move(move):
            messagebox.showinfo("提示", "非法棋步，请重新选择")
            return
        
        # 应用棋步
        self.game_history.append(self.game_state)
        self.game_state = self.game_state.apply_move(move)
        self.current_player = self.game_state.next_player
        
        # 更新状态和棋盘
        self.update_status()
        self.draw_board()
        
        # 检查游戏是否结束
        if self.game_state.is_over():
            self.show_game_result()
            return
        
        # AI 落子
        self.root.after(100, self.ai_move)
    
    def ai_move(self):
        """AI 落子"""
        if not self.game_state or self.game_state.is_over():
            return
        
        # AI 选择棋步
        move = self.ai_agent.select_move(self.game_state)
        
        # 应用棋步
        self.game_history.append(self.game_state)
        self.game_state = self.game_state.apply_move(move)
        self.current_player = self.game_state.next_player
        
        # 更新状态和棋盘
        self.update_status()
        self.draw_board()
        
        # 检查游戏是否结束
        if self.game_state.is_over():
            self.show_game_result()
    
    def undo_move(self):
        """悔棋"""
        if not self.game_history:
            messagebox.showinfo("提示", "没有可悔的棋步")
            return
        
        # 恢复到上一个状态
        self.game_state = self.game_history.pop()
        self.current_player = self.game_state.next_player
        
        # 如果是AI的回合，再悔一步回到用户的回合
        if self.current_player != Player.black:
            if not self.game_history:
                messagebox.showinfo("提示", "没有可悔的棋步")
                return
            self.game_state = self.game_history.pop()
            self.current_player = self.game_state.next_player
        
        # 更新状态和棋盘
        self.update_status()
        self.draw_board()
    
    def update_status(self):
        """更新游戏状态信息"""
        if not self.game_state:
            self.status_var.set("准备开始新游戏")
            return
        
        # 计算提子数
        board = self.game_state.board
        black_stones = 0
        white_stones = 0
        
        for r in range(1, self.board_size + 1):
            for c in range(1, self.board_size + 1):
                stone = board.get(Point(r, c))
                if stone == Player.black:
                    black_stones += 1
                elif stone == Player.white:
                    white_stones += 1
        
        # 构建状态信息
        turn_info = "当前回合: 黑棋" if self.current_player == Player.black else "当前回合: 白棋"
        stones_info = f"黑棋: {black_stones}, 白棋: {white_stones}"
        ai_info = f"AI 类型: {self.ai_type}"
        
        status = f"{turn_info} | {stones_info} | {ai_info}"
        self.status_var.set(status)
    
    def show_game_result(self):
        """显示游戏结果"""
        if not self.game_state or not self.game_state.is_over():
            return
        
        winner = self.game_state.winner()
        if winner == Player.black:
            result = "黑棋获胜！"
        elif winner == Player.white:
            result = "白棋获胜！"
        else:
            result = "平局！"
        
        messagebox.showinfo("游戏结束", result)


def main():
    """主函数"""
    root = tk.Tk()
    app = GoVisualizer(root)
    
    # 处理窗口大小变化
    def on_resize(event):
        app.init_board()
        app.draw_board()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()


if __name__ == "__main__":
    main()