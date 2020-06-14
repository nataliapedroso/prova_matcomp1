# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:32:06 2020

@author: nataliapedroso
"""

"""Código referente a parte A das prova de Matemática Computacional - CAP239"""

##############################################################################
################################################# Importando módulos e pacotes
##############################################################################

import numpy as np
import funcs_prova as funcs
import mfdfa_prova as mfdfa
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import norm, genextreme

##############################################################################
######################################################### Algoritmo cullenfrey
##############################################################################

def cullenfrey(xd,yd,legend, title):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    maior=max(xd)
    polyX1=maior if maior > 4.4 else 4.4
    polyY1=polyX1+1
    polyY2=3/2.*polyX1+3
    y_lim = polyY2 if polyY2 > 10 else 10
    
    x = [0,polyX1,polyX1,0]
    y = [1,polyY1,polyY2,3]
    scale = 1
    poly = Polygon( np.c_[x,y]*scale, facecolor='peachpuff', edgecolor='peachpuff', alpha=0.5)
    ax.add_patch(poly)
    ax.plot(xd,yd, marker="o", c="darkorange", label=legend, linestyle='')
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label ="uniform", marker='^',c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(0, 3, label="normal", marker='*',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 3/2.*np.arange(0,polyX1,0.1)+3, label="gamma", linestyle='-',c='black')
    ax.plot(np.arange(0,polyX1,0.1), 2*np.arange(0,polyX1,0.1)+3, label="lognormal", linestyle='-.',c='black')
    ax.legend()
    ax.set_ylim(y_lim,0)
    ax.set_xlim(-0.2,polyX1)
    plt.xlabel("Skewness²")
    plt.title(title+": Mapa de Cullen-Frey")
    plt.ylabel("Kurtosis")
    plt.savefig(title+legend+"cullenfrey.png")
    plt.show()
    
##############################################################################
################################### MAIN #####################################
##############################################################################

################################################## Cria o input do país Russia
    
fread = open("daily-cases-covid-19.csv", "r")
country = "Russia,"
filename = "russia.png"
title = "Statistcs analysis, country: {}".format(country)

y=[]
date=[]

# Lendo dados do arquivo
for line in fread:
    if "Mar 9" in line and country in line:
        break

for line in fread:
    if country in line and "excl." not in line:
        ctr, code, m, yr, data = line.split(",")
        y.append(int(data))
        date.append(m[1:])
    if country in line and "May 28" in line:
        break
    
########################################################## Plot 1 - histograma
n, bins, patches = plt.hist(y, 60, density=1, facecolor='mediumseagreen', alpha=0.75, label="Normalized data")
plt.title('Histogram')
plt.show()

########################################################## Plot 2 - Cullenfrey
skew = funcs.skewness(y)
kurt = funcs.kurtosis(y)

cullenfrey([skew**2], [kurt+3], "Data", country + " Covid-19")

##################################### Plot 3 - ajuste de uma pdf ao histograma
x = range(len(y))
ymin = min(y)
ymax = max(y)
n = len(y)
ypoints = [(ymin + (i/n) * (ymax-ymin)) for i in range(0, n+1)]

# fit da GEV
mu, sigma = norm.fit(y)
rv_nrm = norm(loc=mu, scale=sigma)
gev_fit = genextreme.fit(y) # estimando GEV
c, loc, scale = gev_fit
mean, var, skew, kurt = genextreme.stats(c, moments='mvsk')
rv_gev = genextreme(c, loc=loc, scale=scale)
gev_pdf = rv_gev.pdf(ypoints) # criando dados a partir do GEV estimado para plotar

plt.title("PDF with data from " + country + "\nmu={0:3.5}, sigma={1:3.5}".format(mu, sigma))

plt.plot(np.arange(min(bins), max(bins)+1, (max(bins) - min(bins))/len(y)), gev_pdf, 'orange', lw=5, alpha=0.6, label='genextreme pdf')
n, bins, patches = plt.hist(y, 60, density=1, facecolor='mediumseagreen', alpha=0.75, label="Normalized data")
plt.ylabel("Probability density")
plt.xlabel("Value")
plt.legend()
plt.savefig("PDF"+filename)
plt.show()

########################### Plot 4 - cálculo do índice espectral alpha via DFA
alfa, xdfa, ydfa, reta = funcs.dfa1d(y, 1)
plt.title(r"Detrended Fluctuation Analysis $\alpha$={0:.3}".format(alfa), fontsize=15)
plt.plot(xdfa, ydfa, marker='o', linestyle='', color="darkorange", label="{0:.3}".format(alfa))
plt.plot(xdfa, reta, color="steelblue")
plt.show()

################################################### Estimando o beta via alpha
beta = 2.*alfa-1.
print("Beta=2*Alpha-1={}".format(beta))

########################################## Plot 5 - espectro de singularidades
plt.figure(figsize=(20, 14))
psi, amax, amin, a0 = mfdfa.makemfdfa(y, True)

############################################# cálculo do delta alpha e A alpha
print("Alpha={0:.3}, 2*Alfa-1={1:.3}, Beta={2:.3}, Delta Alpha={3:.3}, \
      Alpha0={4:.3}, Aalpha={5:.3},"
      .format(alfa, 2*alfa-1, beta, (amax-amin), a0, (a0-amin)/(amax-a0)))
