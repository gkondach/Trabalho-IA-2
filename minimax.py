import numpy as np
import copy

def melhor_jogada(tabuleiro, jogador='X'):
    melhor_valor = -np.inf
    melhor_movimento = None
    
    for i in range(3):
        for j in range(3):
            if tabuleiro[i][j] == ' ':
                novo_tabuleiro = copy.deepcopy(tabuleiro)
                novo_tabuleiro[i][j] = jogador
                valor = minimax(novo_tabuleiro, False, jogador)
                if valor > melhor_valor:
                    melhor_valor = valor
                    melhor_movimento = (i, j)
    return melhor_movimento

def minimax(tabuleiro, maximizando, jogador):
    vencedor = verificar_vencedor(tabuleiro)
    if vencedor == jogador:
        return 1
    elif vencedor == 'Empate':
        return 0
    elif vencedor and vencedor != jogador:
        return -1

    adversario = 'O' if jogador == 'X' else 'X'
    simbolo = jogador if maximizando else adversario
    valores = []

    for i in range(3):
        for j in range(3):
            if tabuleiro[i][j] == ' ':
                novo_tabuleiro = copy.deepcopy(tabuleiro)
                novo_tabuleiro[i][j] = simbolo
                valor = minimax(novo_tabuleiro, not maximizando, jogador)
                valores.append(valor)

    return max(valores) if maximizando else min(valores)

def verificar_vencedor(tabuleiro):
    linhas = list(tabuleiro) + list(tabuleiro.T) + [tabuleiro.diagonal(), np.fliplr(tabuleiro).diagonal()]
    for linha in linhas:
        if np.all(linha == 'X'):
            return 'X'
        elif np.all(linha == 'O'):
            return 'O'
    if np.all(tabuleiro != ' '):
        return 'Empate'
    return None