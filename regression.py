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
    plt.scatter( data[['R']], residuals,color='red',edgecolors='black')
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
    
    return  #residuals, std_error



Xs = data[['M', 'R','ln R','D']]
y = data[['ln pga']]  # Taking the natural logarithm of the PGA values

regression_analysis(Xs,y)
