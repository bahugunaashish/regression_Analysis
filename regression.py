# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:53:00 2023

@author: Ashish Bahuguna 
        IIT Roorkee
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

data = pd.read_csv('H:/Python projects/regression_analysis/regression_analysis/input_with_Depth.csv')  # Replace 'your_data.csv' with the actual file path or data source

# Extract the predictor variables (M, R) and the response variable (PGA)

def regression_analysis(inde_paras,depe_para):
        
    inde_paras = sm.add_constant(inde_paras)
    
    model = sm.OLS(y, inde_paras)
    results = model.fit()
    print(results.summary())
    
    coefficients = results.params
    print('\n========================================\n')
    print('coefficients\n')
    for coe in coefficients:
        print(coe)
    print('\n========================================\n')
    # Calculate the residuals
    residuals = results.resid
    
    # Calculate the standard error of residuals
    std_error = np.std(residuals)
    print('std_error = ', std_error)
    
    # Plot the residuals
    fig1 = plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter( data[['M']], residuals,color='red',edgecolors='black')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axhline(y=std_error, color='green', linestyle='--', label='SE = 0.81')
    plt.axhline(y=-std_error, color='green', linestyle='--')
    plt.xlabel('Magnitude')
    plt.ylabel('Residuals')
    plt.legend()
    plt.tight_layout()
    #plt.title('Residual Plot')
    # plt.show()
    # fig1.savefig('Residual_magnitude.png', dpi = 500)

    plt.subplot(1,2,2)
    plt.scatter( data[['R']], residuals, color='red',edgecolors='black')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axhline(y=std_error, color='green', linestyle='--', label='SE = 0.81')
    plt.axhline(y=-std_error, color='green', linestyle='--')
    plt.xlabel('Epicentre Distance')
    plt.ylabel('Residuals')
    plt.legend()
    plt.tight_layout()
    #plt.title('Residual Plot')
    #plt.show()
    fig1.savefig('Residual.png', dpi = 500)
    
    return  residuals, std_error, coefficients



Xs = data[['M', 'R','ln R','D']]
y = data[['ln pga']]  # Taking the natural logarithm of the PGA values

residuals, std_error, coefficients = regression_analysis(Xs,y)

df3 = data[(data['M'] >= 3.) & (data['M'] < 4.)]
df4 = data[(data['M'] >= 4.) & (data['M'] < 5.)]
df5 = data[(data['M'] >= 5.) & (data['M'] < 6.)]

def gmpe_plot(df, M, D, coefficients):
    pga = np.exp(df[['ln pga']])*100*9.81
    
    # #GMPE plot
   # M =3.
    R = np.arange(0.1, 70., 2.)
    lnR= np.log(R)
   # D = 10
    Y = coefficients[0] + coefficients[1]*M + np.multiply(coefficients[2], R) + np.multiply(coefficients[3],lnR) + coefficients[4]*D
    Y=np.exp(Y)*100*9.81
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(R, Y, color='red', label = 'M = '+str(M))
    plt.scatter(df[['R']], pga, color='red',edgecolors='black', label = 'M = '+str(M))
    plt.yscale("log")
    plt.legend()
    fig.savefig(str(M)+'.png', dpi = 500)
    
M =np.arange(3, 6, 1)
D = 10
for m in M:
    if m == 3:
        gmpe_plot(df3, m, D, coefficients)
    elif m == 4:
        gmpe_plot(df4, m, D, coefficients)
    elif m == 5:
        gmpe_plot(df5, m, D, coefficients)
