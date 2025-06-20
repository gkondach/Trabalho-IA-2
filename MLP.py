import numpy as np

class MLP:
    def __init__(self, input_size=9, hidden_size=18, output_size=9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
        #inicialização dos pesos
        self.w1 = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b1 = np.random.uniform(-1, 1, (hidden_size,))
        self.w2 = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.b2 = np.random.uniform(-1, 1, (output_size,))
        
    def definir_pesos(self, pesos):
        #carrega os pesos de um cromossomo do algoritmo genético
        i, h, o = self.input_size, self.hidden_size, self.output_size
        
        end_w1 = h * i
        end_b1 = end_w1 + h
        end_w2 = end_b1 + o * h
        end_b2 = end_w2 + o
        
        self.w1 = np.array(pesos[:end_w1]).reshape((h, i))
        self.b1 = np.array(pesos[end_w1:end_b1])
        self.w2 = np.array(pesos[end_b1:end_w2]).reshape((o, h))
        self.b2 = np.array(pesos[end_w2:end_b2])
        
    def avancar(self, x):
        h = np.tanh(np.dot(self.w1, x) + self.b1)
        out = np.dot(self.w2, h) + self.b2
        return out
    
    def tabuleiro_entrada(tabuleiro):
        flat = tabuleiro.flatten()
        entrada = np.array([1.0 if c  == 'X' else -1.0 if c == 'O' else 0.0 for c in flat])
        return entrada
    
def agente_mlp(tabuleiro, mlp):
        entrada = MLP.tabuleiro_entrada(tabuleiro)
        saida = mlp.avancar(entrada)
        
        movimentos = [(i, j) for i in range(3) for j in range(3) if tabuleiro[i][j] == ' ']
        if not movimentos:
            return (0, 0)
        
        indices =  [3 * i + j for i, j in movimentos]
        melhor_indice = indices[np.argmax(saida[indices])]
        return divmod(melhor_indice, 3)  # converte o índice linear para coordenadas (linha, coluna)