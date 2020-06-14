# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:05:11 2020

@author: nataliapedroso
"""


##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm, linregress
from scipy import optimize

##############################################################################
################################################## Funções de cálculo estático
##############################################################################

################################################# Função que calcula o momento

def momentum(data,r):
    media=sum(data)/len(data)
    soma=0
    for i in data:
        soma+=(i-media)**r
    soma=soma/len(data)
    return soma

######################################################### Função que normaliza

def normalize(data):
    mini=min(data)
    maxi=max(data)
    normdata=[]
    for i in data:
        normdata.append((i-mini)/(maxi-mini))
    return normdata

############################################### Função que calcula a variância

def variance(data):
    return momentum(data,2)

############################################## Função que calcula a assimetria

def skewness(data):
    n=len(data)
    return (n*(n-1))**(0.5)/(n-2)*(momentum(data,3)/momentum(data,2)**(3./2.))

################################################# Função que calcula a curtose
    
def kurtosis(data):
    n=len(data)
    return (n-1)*(n+1)/((n-2)*(n-3))*(momentum(data,4))/(momentum(data,2)**2)-3*((n-1)**2)/((n-2)*(n-3))

############################### Função que calcula o PSD de uma série temporal

def psd(data):
    # Define um intervalo para realizar o ajuste da reta
    INICIO = 10
    FIM = 200 if len(data)//2 > 200 else len(data)//2
    
    # O vetor com o tempo é o tamanho do número de pontos
    N = len(data)
    tempo = np.arange(len(data))
    # Define a frequência de amostragem
    dt = (tempo[-1] - tempo[0] / (N - 1))
    fs = 1 / dt
    # Calcula o PSD utilizando o MLAB
    power, freqs = mlab.psd(data, Fs = fs, NFFT = N, scale_by_freq = False)
    # Calcula a porcentagem de pontos utilizados na reta de ajuste
    # Seleciona os dados dentro do intervalo de seleção
    xdata = freqs[INICIO:FIM]
    ydata = power[INICIO:FIM]
    # Simula o erro
    yerr = 0.2 * ydata
    # Define uma função para calcular a Lei de Potência
    powerlaw = lambda x, amp, index: amp * (x**index)
    # Converte os dados para o formato LOG
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    # Define a função para realizar o ajuste
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err    
    logyerr = yerr / ydata
    # Calcula a reta de ajuste
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args = (logx, logy, logyerr), full_output = 1)    
    pfinal = out[0]
    index = pfinal[1]
    amp = 10.0 ** pfinal[0]
    # Retorna os valores obtidos
    return freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM

################################################## Função que calcula o DFA 1D 

def dfa1d(timeSeries, grau):

	"""Calcula o DFA 1D (adaptado de Physionet), onde a escala cresce
	de acordo com a variável 'Boxratio'. Retorna o array 'vetoutput', 
	onde a primeira coluna é o log da escala S e a segunda coluna é o
	log da função de flutuação."""

	# 1. A série temporal {Xk} com k = 1, ..., N é integrada na chamada função perfil Y(k)
	x = np.mean(timeSeries)
	timeSeries = timeSeries - x
	yk = np.cumsum(timeSeries)
	tam = len(timeSeries)
	# 2. A série (ou perfil) Y(k) é dividida em N intervalos não sobrepostos de tamanho S
	sf = np.ceil(tam / 4).astype(np.int)
	boxratio = np.power(2.0, 1.0 / 8.0)
	vetoutput = np.zeros(shape = (1,2))
	s = 4
	while s <= sf:        
		serie = yk        
		if np.mod(tam, s) != 0:
			l = s * int(np.trunc(tam/s))
			serie = yk[0:l]			
		t = np.arange(s, len(serie), s)
		v = np.array(np.array_split(serie, t))
		l = len(v)
		x = np.arange(1, s + 1)
		# 3. Calcula-se a variância para cada segmento v = 1,…, n_s:
		p = np.polynomial.polynomial.polyfit(x, v.T, grau)
		yfit = np.polynomial.polynomial.polyval(x, p)
		vetvar = np.var(v - yfit)
        # 4. Calcula-se a função de flutuação DFA como a média das variâncias de cada intervalo
		fs = np.sqrt(np.mean(vetvar))
		vetoutput = np.vstack((vetoutput,[s, fs]))
		# A escala S cresce numa série geométrica
		s = np.ceil(s * boxratio).astype(np.int)
	# Array com o log da escala S e o log da função de flutuação   
	vetoutput = np.log10(vetoutput[1::1,:])
	# Separa as colunas do vetor 'vetoutput'
	x = vetoutput[:,0]
	y = vetoutput[:,1]
	# Regressão linear
	slope, intercept, _, _, _ = linregress(x, y)
	# Calcula a reta de inclinação
	predict_y = intercept + slope * x
	# Calcula o erro
	# Retorna o valor do ALFA, o vetor 'vetoutput', os vetores X e Y,
	# o vetor com os valores da reta de inclinação e o vetor de erros
	return slope, x, y, predict_y

#################################### Função de teste para o mapa de cullenfrey

def teste(N):
    x=range(N)
    y=[]
    for i in x:
        y.append(rnd.normal())
    return x,y

##############################################################################
##############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy.random as rnd
    x,y=teste(8192)
    plt.figure(figsize=(20, 12))
    #Plot da série temporal
    ax1 = plt.subplot(211)
    ax1.set_title("Gaussian RNG", fontsize=18)
    (mu,sigma)=norm.fit(y)
    n, bins, patches = ax1.hist(y, 60, density=1, facecolor='powderblue', alpha=0.75)
    ax1.plot(bins,norm.pdf(bins,mu,sigma), c="black", linestyle='--')
    #Plot e cálculo do DFA
    ax2 = plt.subplot(223)
    slope,xdfa,ydfa,predict_y=dfa1d(y, 1)
    ax2.set_title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(slope, fontsize=15))
    ax2.plot(xdfa,ydfa, marker='o', linestyle='', color="#12355B")
    ax2.plot(xdfa, predict_y, color="#9DACB2")
    #Plot e cálculo do PSD
    freqs, power, xdata, ydata, amp, index, powerlaw, INICIO, FIM = psd(y)
    ax3 = plt.subplot(224)
    ax3.set_title(r"Power Spectrum Density $\beta$={0:.3}".format(index, fontsize=15))
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.plot(freqs, power, '-', color = 'deepskyblue', alpha = 0.7)
    ax3.plot(xdata, ydata, color = "darkblue", alpha = 0.8)
    ax3.axvline(freqs[INICIO], color = "darkblue", linestyle = '--')
    ax3.axvline(freqs[FIM], color = "darkblue", linestyle = '--')    
    ax3.plot(xdata, powerlaw(xdata, amp, index),color="#D65108", linestyle='-', linewidth = 3, label = '$%.4f$' %(index))
    ax2.set_xlabel("log(s)")
    ax2.set_ylabel("log F(s)")
    ax3.set_xlabel("Frequência (Hz)")
    ax3.set_ylabel("Potência")
    plt.show()