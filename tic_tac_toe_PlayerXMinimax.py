import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.array([' '] * 9)  # Tabuleiro 3x3
        self.human_player = 'X'
        self.ai_player = 'O'
        self.current_player = self.human_player  # Humano começa

    def print_board(self):
        print("\n")
        print(f" {self.board[0]} | {self.board[1]} | {self.board[2]} ")
        print("---|---|---")
        print(f" {self.board[3]} | {self.board[4]} | {self.board[5]} ")
        print("---|---|---")
        print(f" {self.board[6]} | {self.board[7]} | {self.board[8]} ")
        print("\n")

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False

    def switch_player(self):
        self.current_player = self.human_player if self.current_player == self.ai_player else self.ai_player

    def check_winner(self):
        # Linhas
        for i in range(0, 9, 3):
            if self.board[i] != ' ' and self.board[i] == self.board[i+1] == self.board[i+2]:
                return self.board[i]
        
        # Colunas
        for i in range(3):
            if self.board[i] != ' ' and self.board[i] == self.board[i+3] == self.board[i+6]:
                return self.board[i]
        
        # Diagonais
        if self.board[0] != ' ' and self.board[0] == self.board[4] == self.board[8]:
            return self.board[0]
        if self.board[2] != ' ' and self.board[2] == self.board[4] == self.board[6]:
            return self.board[2]
        
        # Empate
        if ' ' not in self.board:
            return 'Tie'
        
        return None

    def minimax(self, board, depth, is_maximizing):
        winner = self.check_winner()
        
        if winner == self.ai_player:
            return 10 - depth
        elif winner == self.human_player:
            return depth - 10
        elif winner == 'Tie':
            return 0
        
        if is_maximizing:
            best_score = -float('inf')
            for move in self.available_moves():
                board[move] = self.ai_player
                score = self.minimax(board, depth + 1, False)
                board[move] = ' '
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.available_moves():
                board[move] = self.human_player
                score = self.minimax(board, depth + 1, True)
                board[move] = ' '
                best_score = min(score, best_score)
            return best_score

    def ai_move(self):
        best_score = -float('inf')
        best_move = None
        
        for move in self.available_moves():
            self.board[move] = self.ai_player
            score = self.minimax(self.board, 0, False)
            self.board[move] = ' '
            
            if score > best_score:
                best_score = score
                best_move = move
        
        self.make_move(best_move, self.ai_player)

    def play(self):
        print("Bem-vindo ao Jogo da Velha!")
        print("Você é o 'X' e a IA é o 'O'")
        print("Posições: 0-8 (esquerda para direita, topo para base)\n")
        
        while True:
            self.print_board()
            
            if self.current_player == self.human_player:
                try:
                    move = int(input("Sua vez! Escolha uma posição (0-8): "))
                    if move not in range(9):
                        print("Posição inválida! Escolha entre 0-8.")
                        continue
                    if not self.make_move(move, self.human_player):
                        print("Posição ocupada! Tente novamente.")
                        continue
                except ValueError:
                    print("Entrada inválida! Digite um número.")
                    continue
            else:
                print("Vez da IA...")
                self.ai_move()
            
            winner = self.check_winner()
            if winner:
                self.print_board()
                if winner == 'Tie':
                    print("Empate!")
                elif winner == self.human_player:
                    print("Você venceu!")
                else:
                    print("IA venceu!")
                break
            
            self.switch_player()

if __name__ == "__main__":
    game = TicTacToe()
    game.play()