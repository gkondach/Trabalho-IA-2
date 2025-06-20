import numpy as np
import random
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        return False

    def is_winner(self, player):
        for i in range(3):
            if all(self.board[i] == player):
                return True
        for j in range(3):
            if all(self.board[:, j] == player):
                return True
        if all(np.diag(self.board) == player):
            return True
        if all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return np.count_nonzero(self.board) == 9

    def is_game_over(self):
        return self.is_winner(1) or self.is_winner(2) or self.is_draw()

    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def get_state(self):
        return self.board.flatten()

    def get_state_for_player(self, player):
        transformed = np.zeros(9)
        for i in range(9):
            if self.board.flatten()[i] == player:
                transformed[i] = 1
            elif self.board.flatten()[i] == 3 - player:
                transformed[i] = -1
        return transformed

class Minimax:
    def __init__(self, difficulty="hard", player=2):
        self.difficulty = difficulty
        self.player = player

    def get_move(self, game):
        if self.difficulty == "easy":
            if random.random() < 0.25:
                return self.minimax_move(game)
            else:
                return random.choice(game.available_moves())
        elif self.difficulty == "medium":
            if random.random() < 0.5:
                return self.minimax_move(game)
            else:
                return random.choice(game.available_moves())
        else:
            return self.minimax_move(game)

    def minimax_move(self, game):
        best_score = float('-inf')
        best_move = None
        for move in game.available_moves():
            game.make_move(move[0], move[1])
            score = self.minimax(game, False, game.current_player)
            game.board[move[0]][move[1]] = 0
            game.current_player = 3 - game.current_player
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, game, is_maximizing, current_player):
        if game.is_winner(self.player):
            return 10
        elif game.is_winner(3 - self.player):
            return -10
        elif game.is_draw():
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for move in game.available_moves():
                game.make_move(move[0], move[1])
                score = self.minimax(game, False, 3 - current_player)
                game.board[move[0]][move[1]] = 0
                game.current_player = current_player
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for move in game.available_moves():
                game.make_move(move[0], move[1])
                score = self.minimax(game, True, 3 - current_player)
                game.board[move[0]][move[1]] = 0
                game.current_player = current_player
                best_score = min(score, best_score)
            return best_score

class NeuralNetwork:
    def __init__(self, input_size=9, hidden_size=18, output_size=9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias1 = np.random.uniform(-1, 1, hidden_size)
        self.weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias2 = np.random.uniform(-1, 1, output_size)

    def forward(self, x):
        self.layer1 = np.dot(x, self.weights1) + self.bias1
        self.layer1_act = self.relu(self.layer1)
        self.output = np.dot(self.layer1_act, self.weights2) + self.bias2
        return self.output

    def relu(self, x):
        return np.maximum(0, x)

    def set_weights(self, chromosome):
        idx = 0
        self.weights1 = chromosome[idx:9*18].reshape((9, 18))
        idx += 9*18
        self.bias1 = chromosome[idx:idx+18]
        idx += 18
        self.weights2 = chromosome[idx:idx+18*9].reshape((18, 9))
        idx += 18*9
        self.bias2 = chromosome[idx:idx+9]

    def get_weights(self):
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1,
            self.weights2.flatten(),
            self.bias2
        ])

    def predict(self, state, player):
        transformed_state = self.transform_state(state, player)
        output = self.forward(transformed_state)
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        return probabilities

    def transform_state(self, state, player):
        transformed = np.zeros(9)
        for i in range(9):
            if state[i] == player:
                transformed[i] = 1
            elif state[i] == 3 - player:
                transformed[i] = -1
        return transformed

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.05, crossover_rate=0.8, elitism_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate

    def initialize_population(self, chromosome_length):
        return [np.random.uniform(-1, 1, chromosome_length) for _ in range(self.population_size)]

    def evaluate_fitness(self, population, game, minimax, mode):
        fitness_scores = []
        for chromosome in population:
            net = NeuralNetwork()
            net.set_weights(chromosome)
            score = 0
            for _ in range(5):
                result = self.play_game(net, game, minimax, mode)
                score += result
            fitness_scores.append(score)
        return fitness_scores

    def play_game(self, net, game, minimax, mode):
        game.reset()
        minimax.difficulty = mode
        current_player = 1
        
        while not game.is_game_over():
            if current_player == 1:
                state = game.get_state()
                probabilities = net.predict(state, 1)
                moves = game.available_moves()
                
                if not moves:
                    break
                    
                move_probs = []
                for move in moves:
                    idx = move[0] * 3 + move[1]
                    move_probs.append(probabilities[idx])
                    
                best_idx = np.argmax(move_probs)
                chosen_move = moves[best_idx]
                
                if not game.make_move(chosen_move[0], chosen_move[1]):
                    return -20 #jogada errada
            else:
                move = minimax.get_move(game)
                game.make_move(move[0], move[1])
                
            current_player = 3 - current_player

        if game.is_winner(1):
            return 20 # vitoria
        elif game.is_winner(2):
            return -10 # derrota
        else:
            return 15 # empate

    def select_parents(self, population, fitness_scores):
        elite_size = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        parents = [population[i] for i in elite_indices]
        
        while len(parents) < self.population_size:
            tournament = random.sample(range(len(population)), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner_idx = tournament[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
            
        return parents

    def crossover(self, parent1, parent2):
        alpha = 0.5
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            cmin = min(parent1[i], parent2[i]) - alpha * abs(parent1[i] - parent2[i])
            cmax = max(parent1[i], parent2[i]) + alpha * abs(parent1[i] - parent2[i])
            child[i] = random.uniform(cmin, cmax)
        return child

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] += random.gauss(0, 0.1)
                chromosome[i] = np.clip(chromosome[i], -1, 1)
        return chromosome

    def evolve(self, population, fitness_scores):
        parents = self.select_parents(population, fitness_scores)
        next_population = parents[:int(self.elitism_rate * self.population_size)]
        
        while len(next_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = self.mutate(child)
            next_population.append(child)
            
        return next_population

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Jogo da Velha - IA")
        self.game = TicTacToe()
        self.minimax = Minimax()
        self.neural_net = NeuralNetwork()
        self.ga = GeneticAlgorithm()
        self.training = False
        self.current_mode = None
        self.best_net = None
        self.fitness_history = []
        self.avg_fitness_history = []

        # Configuração da interface
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(padx=10, pady=10)

        # Botões do menu
        tk.Button(self.main_frame, text="Jogar contra Minimax", 
                 command=lambda: self.setup_game('minimax')).pack(fill=tk.X, pady=5)
        tk.Button(self.main_frame, text="Treinar Rede Neural", 
                 command=lambda: self.setup_game('train')).pack(fill=tk.X, pady=5)
        tk.Button(self.main_frame, text="Jogar contra Rede Treinada", 
                 command=lambda: self.setup_game('neural_net')).pack(fill=tk.X, pady=5)

        # Frame do tabuleiro
        self.board_frame = tk.Frame(self.master)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        
        # Frame de treinamento
        self.training_frame = tk.Frame(self.master)
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.training_frame)
        
        # Barra de progresso
        self.progress = ttk.Progressbar(self.master, orient="horizontal", 
                                       length=300, mode="determinate")
        
        # Dificuldade
        self.difficulty_frame = tk.Frame(self.master)
        self.difficulty_var = tk.StringVar(value="hard")

    def setup_game(self, mode):
        self.current_mode = mode
        self.game.reset()
        
        # Limpar frames anteriores
        self.clear_frames()
        
        # Configurar tabuleiro
        self.board_frame.pack(pady=10)
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Button(
                    self.board_frame, text="", width=10, height=3,
                    command=lambda row=i, col=j: self.on_click(row, col),
                    font=('Arial', 14))
                self.buttons[i][j].grid(row=i, column=j, padx=2, pady=2)
        
        # Configurações específicas de modo
        if mode == 'minimax':
            self.setup_minimax_mode()
        elif mode == 'train':
            self.setup_train_mode()
        elif mode == 'neural_net':
            self.setup_neural_net_mode()
            
        self.update_board()

    def clear_frames(self):
        self.board_frame.pack_forget()
        self.training_frame.pack_forget()
        self.progress.pack_forget()
        self.difficulty_frame.pack_forget()
        for widget in self.difficulty_frame.winfo_children():
            widget.destroy()

    def setup_minimax_mode(self):
        self.difficulty_frame.pack(pady=5)
        tk.Label(self.difficulty_frame, text="Dificuldade:").pack(side=tk.LEFT)
        
        difficulties = [("Fácil", "easy"), ("Médio", "medium"), ("Difícil", "hard")]
        for text, mode in difficulties:
            rb = tk.Radiobutton(
                self.difficulty_frame, text=text, variable=self.difficulty_var,
                value=mode, command=self.update_difficulty)
            rb.pack(side=tk.LEFT, padx=5)

    def setup_train_mode(self):
        self.training_frame.pack(pady=10)
        self.canvas.get_tk_widget().pack()
        self.progress.pack(pady=10)
        
        tk.Button(
            self.master, text="Iniciar Treinamento", 
            command=self.start_training, font=('Arial', 12)
        ).pack(pady=10)

    def setup_neural_net_mode(self):
        if self.best_net is None:
            messagebox.showinfo("Aviso", "Treine a rede neural primeiro!")
            return

    def update_difficulty(self):
        self.minimax.difficulty = self.difficulty_var.get()

    def on_click(self, row, col):
        if self.current_mode == 'minimax':
            self.handle_minimax_move(row, col)
        elif self.current_mode == 'neural_net':
            self.handle_neural_net_move(row, col)

    def handle_minimax_move(self, row, col):
        if self.game.make_move(row, col):
            self.update_board()
            if self.game.is_game_over():
                self.show_game_result()
                return
            self.master.after(500, self.make_minimax_move)

    def handle_neural_net_move(self, row, col):
        if self.game.make_move(row, col):
            self.update_board()
            if self.game.is_game_over():
                self.show_game_result()
                return
            self.master.after(500, self.make_neural_net_move)

    def make_minimax_move(self):
        self.minimax.difficulty = self.difficulty_var.get()
        move = self.minimax.get_move(self.game)
        self.game.make_move(move[0], move[1])
        self.update_board()
        if self.game.is_game_over():
            self.show_game_result()

    def make_neural_net_move(self):
        state = self.game.get_state()
        probabilities = self.best_net.predict(state, 2)
        moves = self.game.available_moves()
        
        if moves:
            move_probs = []
            for move in moves:
                idx = move[0] * 3 + move[1]
                move_probs.append(probabilities[idx])
                
            best_idx = np.argmax(move_probs)
            chosen_move = moves[best_idx]
            self.game.make_move(chosen_move[0], chosen_move[1])
            self.update_board()
            if self.game.is_game_over():
                self.show_game_result()

    def update_board(self):
        for i in range(3):
            for j in range(3):
                if self.game.board[i][j] == 1:
                    self.buttons[i][j].config(text="X", fg="blue", state=tk.DISABLED)
                elif self.game.board[i][j] == 2:
                    self.buttons[i][j].config(text="O", fg="red", state=tk.DISABLED)
                else:
                    self.buttons[i][j].config(text="", state=tk.NORMAL)

    def show_game_result(self):
        if self.game.is_winner(1):
            messagebox.showinfo("Fim de jogo", "Jogador X venceu!")
        elif self.game.is_winner(2):
            messagebox.showinfo("Fim de jogo", "Jogador O venceu!")
        else:
            messagebox.showinfo("Fim de jogo", "Empate!")
        self.game.reset()
        self.update_board()

    def start_training(self):
        self.training = True
        self.fitness_history = []
        self.avg_fitness_history = []
        
        # Configurar GA
        net = NeuralNetwork()
        chromosome_length = len(net.get_weights())
        population = self.ga.initialize_population(chromosome_length)
        
        # Modos de treinamento progressivo
        modes = ["easy"] * 20 + ["medium"] * 50 + ["hard"] * 30
        num_generations = len(modes)
        self.progress["maximum"] = num_generations
        
        # Loop de treinamento
        for gen in range(num_generations):
            mode = modes[gen]
            fitness_scores = self.ga.evaluate_fitness(population, self.game, self.minimax, mode)
            
            # Registrar métricas
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Atualizar gráfico
            self.update_plot(gen+1)
            
            # Evoluir população
            population = self.ga.evolve(population, fitness_scores)
            
            # Atualizar barra de progresso
            self.progress["value"] = gen+1
            self.master.update()
        
        # Selecionar melhor rede
        best_idx = np.argmax(fitness_scores)
        self.best_net = NeuralNetwork()
        self.best_net.set_weights(population[best_idx])
        
        messagebox.showinfo("Treinamento", "Treinamento concluído com sucesso!")
        self.training = False

    def update_plot(self, generation):
        self.ax.clear()
        self.ax.plot(range(1, generation+1), self.fitness_history, 'b-', label='Melhor Fitness')
        self.ax.plot(range(1, generation+1), self.avg_fitness_history, 'r-', label='Fitness Médio')
        self.ax.set_title(f"Evolução do Fitness (Geração {generation})")
        self.ax.set_xlabel("Geração")
        self.ax.set_ylabel("Fitness")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()