import sys
from MLP import MLP, agente_mlp
from TicTacToe import executar_partida, agente_aleatorio, TicTacToe
from minimax import melhor_jogada
from Algoritmo_Genetico import avaliar_rede_final, gerar_populacao, evoluir, FASES, avaliacao_individual
from Algoritmo_Genetico import base
from multiprocessing import Pool
import numpy as np
import random

def jogador_humano(tabuleiro):
    print("\nSeu turno! Tabuleiro atual:")
    jogo = TicTacToe()
    jogo.tabuleiro = tabuleiro.copy()
    jogo.printar_tabuleiro()

    while True:
        try:
            entrada = input("Digite sua jogada (linha,coluna): ")
            linha, coluna = map(int, entrada.strip().split(","))
            if tabuleiro[linha][coluna] == ' ':
                return linha, coluna
            else:
                print("Posição ocupada. Tente novamente.")
        except:
            print("Entrada inválida. Use formato linha,coluna (ex: 0,1)")

def jogar_contra_algoritmo_genetico():
    print("\n--- Jogar contra Rede MLP treinada ---")
    rede = MLP()
    try:
        rede.definir_pesos(np.load("melhor_pesos.npy"))
        resultado = executar_partida(jogador_humano, lambda t: agente_mlp(t, rede), show=True)
        print(f"Resultado final: {resultado}")
    except FileNotFoundError:
        print("\nRede ainda não foi treinada. Execute o treinamento primeiro.")

def jogar_contra_minimax():
    print("\nEscolha dificuldade: 1) Fácil  2) Médio  3) Difícil")
    escolha = input("> ").strip()
    if escolha == '1':
        def bot(tabuleiro):
            return melhor_jogada(tabuleiro, 'O') if random.random() < 0.25 else agente_aleatorio(tabuleiro)
    elif escolha == '2':
        def bot(tabuleiro):
            return melhor_jogada(tabuleiro, 'O') if random.random() < 0.5 else agente_aleatorio(tabuleiro)
    else:
        def bot(tabuleiro):
            return melhor_jogada(tabuleiro, 'O')

    resultado = executar_partida(jogador_humano, bot, show=True)
    print(f"Resultado final: {resultado}")

def treinar_algoritmo_genetico():
    print("\n--- Treinando Rede MLP com Algoritmo Genético ---")
    populacao = gerar_populacao(base)

    for adversario, num_geracoes in FASES:
        print(f"\n=== Treinando contra: {adversario.__name__} ===")
        for g in range(num_geracoes):
            print(f"Geração {g+1} ({adversario.__name__})")
            with Pool() as pool:
                args = [(c, base, adversario) for c in populacao]
                fitnesses = pool.map(avaliacao_individual, args)
            print(f"Melhor fitness: {max(fitnesses)} | Média: {np.mean(fitnesses):.2f}")
            populacao = evoluir(populacao, fitnesses)

    melhor = max(zip(populacao, fitnesses), key=lambda x: x[1])[0]
    base.definir_pesos(melhor)
    np.save("melhor_pesos.npy", melhor)

    avaliar_rede_final(base)

def menu():
    while True:
        print("\n===== MENU PRINCIPAL =====")
        print("1) Jogar contra Rede MLP treinada")
        print("2) Jogar contra Minimax (fácil/médio/difícil)")
        print("3) Treinar Rede com Algoritmo Genético")
        print("4) Sair")
        escolha = input("Escolha uma opção: ").strip()

        if escolha == '1':
            jogar_contra_algoritmo_genetico()
        elif escolha == '2':
            jogar_contra_minimax()
        elif escolha == '3':
            treinar_algoritmo_genetico()
        elif escolha == '4':
            print("Encerrando.")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu()