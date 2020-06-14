# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:45:41 2020

@author: nataliapedroso
"""

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
##############################################################################

fread = open("daily-cases-covid-19.csv", "r")   # Abrindo o arquivo
p = [[0.5, 0.45, 0.05], [0.7, 0.25, 0.05]]      # Fazendo p como lista para ficar mais fácil de trocar
pind = 0                                        # Indice
vals = [[1, 3, 5], [2, 4, 6]]                   # Valores usados para multiplicar nk
y = []                                          # Onde o dado será armazenado
date = []                                       # Usado para melhorar o gráfico

country = 'Russia,'                             # País escolhido

# Lendo dados do arquivo (começando dia 2 de março)
for line in fread:
    if "Mar 2" in line and country in line and "excl." not in line:
        break

# Parando em 28 de maio
for line in fread:
    if country in line and "excl." not in line:
        ctr, code, m, yr, data = line.split(",")
        y.append(int(data))
        date.append(m[1:])
    if country in line and "May 28" in line:
        break

country=country[:-1]
fread.close()           # Fechando o arquivo

meandays = 7            # Quantos dias a média usará

xticks = []     # Melhora o gráfico, adicionando a data no eixo x
for i in range(meandays, len(date)):
    if i % meandays == 0:
        xticks.append(date[i])

# Aqui as médias são calculadas fazendo uma sublista da lista original e somando
# seus valores. O número de dias será determinado pela variável meandays e então
# G será calculado por essa média, usando o valor de um certo dia e a soma dos
# meandays anteriores. Depois, Nmin e Nmax são calculados pelo modelo. O valor
# de Nguess é determinado pela média entre Nmin e Nmax. Além disso,  
# deltank = (meandays - dia atual)/dia atual.
        
for pind in range(len(p)):
    # Inicializando as listas para armazenar os valores dos modelos
    Nmin = []       # lista Nmin
    Nmax = []       # lista Nmax 
    Nguess = []     # média entra Nmax e Nmin 
    Nk7 = []        # armazena todas as médias de meandays
    g = []          # armazena os valores de g
    deltank = []    # armazena delta NK
    for i in range(meandays, len(y)):
        Nk7.append((sum(y[i-meandays:i]))/meandays)
        if y[i] < Nk7[-1]:
            g.append((y[i]/Nk7[-1]))
            w = [1, 1]
        else:
            g.append((Nk7[-1]/y[i]))
            w = [1, 1]
        n = np.dot(p[pind], y[i])
        Nmin.append(g[-1]*np.dot(n, vals[0]))
        Nmax.append(g[-1]*np.dot(n, vals[1]))
        Nguess.append((w[0]*Nmin[-1]+w[1]*Nmax[-1])/sum(w))
        if y[i] != 0:
            deltank.append((Nk7[-1]-y[i])/y[i])
        else:
            deltank.append(np.nan)
    
    # Calculando deltag para calcular s e plotar
    deltag = [0]
    for i in range(1, len(g)):
        g0 = g[i-1]
        if g0 < g[i]:
            deltag.append(g0-g[i] - (1-g[i])**2)
        else:
            deltag.append(g0-g[i] + (1-g0)**2)
    
    deltag = np.array(deltag)
    deltank = np.array(deltank)
    s = (2*deltag + deltank)/3

    # Plotagem.
    # Analisando os modelos comparando os gráficos dos dados que já temos com 
    # os gráficos das variáveis Nmin, Nmax e Nguess.

    plt.title("Graph with the data and the mean of 7 days for each data")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y)-meandays), y[meandays:], label="Data")
    plt.plot(range(len(Nk7)), Nk7, label="{} days means".format(meandays))
    plt.legend()
    plt.savefig("{}meananddata.png".format(country))
    plt.show()

    plt.title("Original Data with predictions")
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y)-meandays), y[meandays:], label="Data")
    plt.plot(range(len(Nguess)), Nguess, label="Predict")
    plt.xticks(np.arange(80, step=meandays), xticks, rotation=45)
    plt.plot(range(len(Nmin)), Nmin, label="Nmin")
    plt.plot(range(len(Nmax)), Nmax, label="Nmax")
    plt.legend()
    plt.savefig("{}originaldata.png".format(country))
    plt.show()

    # Plotando os valores de g
    g = np.array(g)
    plt.figure(figsize=(20, 10))
    meang = abs(sum(g)/len(g)-g)
    plt.title("Values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.errorbar(range(len(g)), g, yerr=meang, xerr=0, hold=True, ecolor='k', fmt='none', label='data', elinewidth=0.5, capsize=1)
    plt.plot(range(len(g)), g, 'o-')
    plt.savefig("{}originalg.png".format(country))
    plt.show()

    # Plotando os valores de s
    s = np.array(s)
    plt.figure(figsize=(20, 10))
    means = abs(sum(s)/len(s)-s)
    plt.title("Values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.errorbar(range(len(s)), s, yerr=meang, xerr=0, hold=True, ecolor='k', fmt='none', label='data', elinewidth=0.5, capsize=1)
    plt.plot(range(len(s)), s, 'o-')
    plt.savefig("{}originals.png".format(country))
    plt.show()


    # Fazendo uma previsão (sem os dados originais)
    preddays = 20               # Quantos dias serão previstos
    predictNmin = [Nmin[-1]]    # Previsão de Nmin
    predictNmax = [Nmax[-1]]    # Previsão de Nmax
    predictg = []               # g calculado na previsão
    predictNmed = y[-meandays-1:]  # Começando a previsão com dados reais
    predictNk7 = []             # Lista menadays para a previsão
    predictdeltank = []         # Lista Delta NK para a previsão
    for i in range(meandays, preddays+meandays):
        predictNk7.append(sum(predictNmed[i-meandays:i])/meandays)
        if predictNmed[i] < predictNk7[-1]:
            predictg.append((predictNmed[i]/predictNk7[-1]))
            w = [1, 1]
        else:
            predictg.append((predictNk7[-1]/predictNmed[i]))
            w = [1, 1]
        n = np.dot(p[pind], predictNmed[-1])
        predictNmin.append(predictg[-1]*np.dot(n, vals[0]))
        predictNmax.append(predictg[-1]*np.dot(n, vals[1]))
        predictNmed.append((w[0]*predictNmin[-1]+w[1]*predictNmax[-1])/sum(w))
        predictdeltank.append((predictNk7[-1]-predictNmed[-1])/predictNmed[-1])
    
    plt.title("Plot showing the prediction for the next {} days,\n p={}, {}, {}".format(preddays, p[pind][0],p[pind][1],p[pind][2]))
    plt.ylabel("New Cases")
    plt.xlabel("Days")
    plt.plot(range(len(y)-meandays), y[meandays:], label="Data")
    plt.plot(range(len(Nguess)), Nguess, label="Nmed", c="orange")
    plt.plot(range(len(y)-meandays-1, len(y)+preddays-meandays), predictNmed[meandays:], c="orange", linestyle='--', label="Predict Nmed")
    plt.legend()
    plt.savefig("{}predictmeananddata{}.png".format(country,pind))
    plt.show()
    
    plt.title("Predict values of g")
    plt.xlabel("Day")
    plt.ylabel("g")
    plt.plot(range(len(g)), g, c="orange", label="g from data")
    plt.plot(range(len(g)-1, len(g)+preddays-1), predictg, c="orange", linestyle='--', label="Generated g")
    plt.legend()
    plt.savefig("{}predictg{}.png".format(country,pind))
    plt.show()
    
    predictdeltag = [0]
    for i in range(1, len(predictg)):
        g0 = predictg[i-1]
        if g0 < predictg[i]:
            predictdeltag.append(g0-predictg[i] - (1-predictg[i])**2)
        else:
            predictdeltag.append(g0-predictg[i] + (1-g0)**2)
    
    predictdeltag = np.array(predictdeltag)
    predictdeltank = np.array(predictdeltank)
    predicts = (2*predictdeltag + predictdeltank)/3
    
    plt.title("Predict values of s")
    plt.xlabel("Day")
    plt.ylabel("s")
    plt.plot(range(len(s)), s, c="limegreen", label="g from data")
    plt.plot(range(len(s)-1, len(s)+preddays-1), predicts, c="limegreen", linestyle='--', label="Generated s")
    plt.legend()
    plt.savefig("{}predicts{}.png".format(country,pind))
    plt.show()