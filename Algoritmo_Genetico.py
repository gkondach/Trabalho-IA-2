import numpy as np
from MLP import MLP, agente_mlp
import random
import time
from TicTacToe import executar_partida, agente_aleatorio
from minimax import melhor_jogada
from multiprocessing import Pool
import matplotlib.pyplot as plt

POPULACAO = 10
PARTIDAS = 3
GERACOES = 10

def gerar_cromossomo(size):
    return np.random.uniform(-1, 1, size)

def gerar_populacao(modelo_mlp):
    numero_pesos = modelo_mlp.w1.size + modelo_mlp.b1.size + modelo_mlp.w2.size + modelo_mlp.b2.size
    return [gerar_cromossomo(numero_pesos) for _ in range(POPULACAO)]

def minimax_facil(tabuleiro):
    return melhor_jogada(tabuleiro, jogador='O') if random.random() < 0.25 else agente_aleatorio(tabuleiro)

def minimax_medio(tabuleiro):
    return melhor_jogada(tabuleiro, jogador='O') if random.random() < 0.5 else agente_aleatorio(tabuleiro)

def minimax_dificil(tabuleiro):
    return melhor_jogada(tabuleiro, jogador='O')
    
def avaliacao(cromossomo, modelo, adversario):
    modelo.definir_pesos(cromossomo)
    pontuacao =  0

    for _ in range(PARTIDAS):
        resultado = executar_partida(
            lambda t: agente_mlp(t, modelo),
            adversario,
            show=False
        )
        if resultado == 'X':
            pontuacao += 3
        elif resultado == 'O':
            pontuacao += 1
        elif "invalida" in resultado:
            pontuacao -= 5
    return pontuacao

def avaliacao_individual(args):
    cromossomo, modelo_base, adversario = args
    modelo = MLP()
    modelo.definir_pesos(cromossomo)
    pontuacao = 0

    for _ in range(PARTIDAS):
        resultado = executar_partida(
            lambda t: agente_mlp(t, modelo),
            adversario,
            show=False
        )
        if resultado == 'X':
            pontuacao += 3
        elif resultado == 'O':
            pontuacao += 1
        elif "invalida" in resultado:
            pontuacao -= 5

    return pontuacao

def avaliar_rede_final(modelo, n=50):
    vitorias = 0
    empates = 0
    derrotas = 0

    for _ in range(n):
        resultado = executar_partida(
            lambda t: agente_mlp(t, modelo),
            agente_aleatorio,
            show=False
        )
        if resultado == 'X':
            vitorias += 1
        elif resultado == 'O':
            derrotas += 1
        else:
            empates += 1

    print(f"\nAvaliação Final após {n} jogos:")
    print(f"Vitórias: {vitorias} | Empates: {empates} | Derrotas: {derrotas}")
    print(f"Acurácia (vitórias + 0.5 * empates): {(vitorias + 0.5 * empates) / n:.2%}")


def torneio(populacao, fitnesses, k=3):
    selecionados = []
    for _ in range(len(populacao)):
        aspirantes = random.sample(list(zip(populacao, fitnesses)), k)
        vencedor = max(aspirantes, key=lambda x: x[1])
        selecionados.append(vencedor[0])
    return selecionados
    
def elitismo(populacao, fitnesses, n_elite=2):
    pares = list(zip(populacao, fitnesses))
    pares.sort(key=lambda x: x[1], reverse=True)
    elite = [ind for ind, fit in pares[:n_elite]]
    return elite

def blend_crossover(pai1, pai2, alpha=0.5):
    filho1 = np.empty_like(pai1)
    filho2 = np.empty_like(pai2)
    for i in range(len(pai1)):
        d = abs(pai1[i] - pai2[i])
        minimo = min(pai1[i], pai2[i]) - alpha * d
        maximo = max(pai1[i], pai2[i]) + alpha * d
        filho1[i] = np.random.uniform(minimo, maximo)
        filho2[i] = np.random.uniform(minimo, maximo)
    return filho1, filho2

def mutacao(cromossomo, taxa_mutacao=0.01, sigma=0.1):
    for i in range(len(cromossomo)):
        if random.random() < taxa_mutacao:
            cromossomo[i] += np.random.normal(0, sigma)
    return cromossomo

def evoluir(populacao, fitnesses, n_elite=2, taxa_mutacao=0.01):
    elite = elitismo(populacao, fitnesses, n_elite)
    selecionados = torneio(populacao, fitnesses)
    
    nova_populacao = elite.copy()
    
    while len(nova_populacao) < len(populacao):
        pai1, pai2 = random.sample(selecionados, 2)
        filho1, filho2 = blend_crossover(pai1, pai2)
        nova_populacao.append(mutacao(filho1, taxa_mutacao))
        if len(nova_populacao) < len(populacao):
            nova_populacao.append(mutacao(filho2, taxa_mutacao))
            
    return nova_populacao[:len(populacao)]

FASES = [
    (minimax_facil, 5),    # 5 gerações contra fácil
    (minimax_medio, 5),    # depois 5 contra médio
    (minimax_dificil, 5),  # depois 5 contra difícil
]
base = MLP()

if __name__ == "__main__":
    populacao = gerar_populacao(base)
    historico_melhor = []
    historico_media = []

    for adversario, num_geracoes in FASES:
        print(f"\n=== Treinando contra modo: {adversario.__name__} ===")
        for g in range(num_geracoes):
            print(f"\nGeração {g+1} ({adversario.__name__})")
            with Pool() as pool:
                args = [(c, base, adversario) for c in populacao]
                fitnesses = pool.map(avaliacao_individual, args)
            melhor_fit = max(fitnesses)
            media_fit = np.mean(fitnesses)
            historico_melhor.append(melhor_fit)
            historico_media.append(media_fit)
            print(f"Melhor fitness: {melhor_fit} - Média: {media_fit:.2f}")

            populacao = evoluir(populacao, fitnesses)
            
    plt.figure(figsize=(10, 5))
    plt.plot(historico_melhor, label='Melhor Fitness', marker='o')
    plt.plot(historico_media, label='Média de Fitness', marker='x')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Treinamento da Rede MLP')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Avaliação final
    melhor = max(zip(populacao, fitnesses), key=lambda x: x[1])[0]
    base.definir_pesos(melhor)
    
    melhor = max(zip(populacao, fitnesses), key=lambda x: x[1])[0]
    base.definir_pesos(melhor)

    avaliar_rede_final(base)

    print("\n--- Avaliação final contra aleatório ---")
    resultado = executar_partida(
        lambda t: agente_mlp(t, base),
        agente_aleatorio,
        show=True
    )
    print(f"\nDesempenho final: {resultado}")
    inicio = time.time()
    print(f"\nTempo total: {time.time() - inicio:.2f} segundos")
