import numpy as np
import scipy.stats as stat
import sklearn.linear_model as lm
import matplotlib.pyplot as plt


eps = .0001
tau = .01

N_genes = 100
N_population = 100000

def runExperiment(seed):
    rng = np.random.default_rng(seed)
    betas_true = rng.normal(size=N_genes)
    genes_latent = rng.multivariate_normal(np.zeros(N_genes),np.eye(N_genes),
                                            size=N_population)

    quants = np.quantile(genes_latent,1-eps,axis=0)

    genes = 1*(genes_latent>quants)

    liability_latent = np.dot(genes,betas_true)
    liability_latent_scaled = liability_latent/np.std(liability_latent)
    liability = np.sqrt(1-.2)*liability_latent_scaled + np.sqrt(.2)*rng.normal(size=N_population)

    quant_liab = np.quantile(liability,1-tau)
    has_disease = 1.*(liability>quant_liab)

    model = lm.LogisticRegression()
    model.fit(genes,has_disease)
    return betas_true,model.coef_

trial1_true,trial1_learned = runExperiment(1993)
trial2_true,trial2_learned = runExperiment(1990)
trial3_true,trial3_learned = runExperiment(2021)
trial4_true,trial4_learned = runExperiment(2022)
trial5_true,trial5_learned = runExperiment(2023)
trial6_true,trial6_learned = runExperiment(2024)
trial7_true,trial7_learned = runExperiment(2025)
fs1 = 20
fs2 = 24
fs3 = 16
plt.scatter(trial1_true,trial1_learned,c='deeppink',alpha=.8,label='Trial 1')
plt.scatter(trial2_true,trial2_learned,c='dodgerblue',alpha=.8,label='Trial 2')
plt.scatter(trial3_true,trial3_learned,c='gold',alpha=.8,label='Trial 3')
plt.scatter(trial4_true,trial4_learned,c='lime',alpha=.8,label='Trial 4')
plt.scatter(trial5_true,trial5_learned,c='indigo',alpha=.8,label='Trial 5')
plt.scatter(trial6_true,trial6_learned,c='saddlebrown',alpha=.8,label='Trial 6')
plt.scatter(trial7_true,trial7_learned,c='aqua',alpha=.8,label='Trial 7')
plt.xlabel('Latent effect on liability',fontsize=fs1)
plt.ylabel('Regression Coefficient',fontsize=fs1)
plt.title('True vs estimated coefficients .0001',fontsize=fs2)
plt.legend(loc='upper left',fontsize=fs3)
plt.show()
