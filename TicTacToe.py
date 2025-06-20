import numpy as np
import random
from MLP import MLP, agente_mlp

class TicTacToe:
    def __init__(self):
        self.tabuleiro = np.full((3, 3), ' ')
        self.jogador_atual = 'X'  # Rede Neural começa jogando

    def printar_tabuleiro(self):
        for linha in self.tabuleiro:
            print('|'.join(linha))
            print('-' * 5)

    def realizar_movimento(self, linha, coluna, jogador):
        if self.tabuleiro[linha][coluna] != ' ':
            return False  # jogada inválida
        self.tabuleiro[linha][coluna] = jogador
        return True

    def movimentos_disponiveis(self):
        return [(i, j) for i in range(3) for j in range(3) if self.tabuleiro[i][j] == ' ']

    def reset(self):
        self.tabuleiro = np.full((3, 3), ' ')
        self.jogador_atual = 'X'

    def verificar_vencedor(self):
        t = self.tabuleiro
        linhas = list(t) + list(t.T) + [t.diagonal(), np.fliplr(t).diagonal()]
        for linha in linhas:
            if np.all(linha == 'X'):
                return 'X'
            elif np.all(linha == 'O'):
                return 'O'
        if np.all(t != ' '):
            return 'Empate'
        return None  # jogo continua
    
def agente_aleatorio(board):
    disponiveis = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
    return random.choice(disponiveis)

def executar_partida(agente_X, agente_O, show=False):
    partida = TicTacToe()
    while True:
        if partida.jogador_atual == 'X':
            linha, coluna = agente_X(partida.tabuleiro.copy())
        else:
            linha, coluna = agente_O(partida.tabuleiro.copy())

        valido = partida.realizar_movimento(linha, coluna, partida.jogador_atual)
        if not valido:
            return f"Jogada inválida pelo jogador {partida.jogador_atual}"

        if show:
            partida.printar_tabuleiro()
            print()

        vencedor = partida.verificar_vencedor()
        if vencedor:
            return vencedor

        partida.jogador_atual = 'O' if partida.jogador_atual == 'X' else 'X'

if __name__ == "__main__":
    rede = MLP()
    print("\n--- Jogo: Rede MLP (X) vs Aleatório (O) ---")
    resultado = executar_partida(
        lambda board: agente_mlp(board, rede),
        agente_aleatorio,
        show=True
    )
    print(f"Resultado: {resultado}")