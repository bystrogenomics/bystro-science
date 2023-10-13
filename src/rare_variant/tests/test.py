import numpy as np
import pickle

from simulate_multivariate import simulate_rare_variate_multivariate

n_diseases = 6
prevalences = 0.0001 * np.ones(n_diseases)
population_size = 50000
number_of_genes = 1000
mutation_rate = 0.0001

relevant_gene_frequencies = 0.1 * np.ones(n_diseases)
relevant_gene_correlation = np.eye(n_diseases)

correlations = 0.95
relevant_gene_correlation[0, 1] = correlations
relevant_gene_correlation[1, 0] = correlations
relevant_gene_correlation[2, 3] = correlations
relevant_gene_correlation[3, 2] = correlations
relevant_gene_correlation[4, 5] = correlations
relevant_gene_correlation[5, 4] = correlations

heritabilities = 0.8 * np.ones(n_diseases)
rho = 0.5
heritability_correlation = np.eye(n_diseases)
heritability_correlation[0, 1] = rho
heritability_correlation[1, 0] = rho
heritability_correlation[2, 3] = rho
heritability_correlation[3, 2] = rho
heritability_correlation[4, 5] = rho
heritability_correlation[5, 4] = rho

simulate_rare_variate_multivariate(
    prevalences,
    population_size,
    number_of_genes,
    mutation_rate,
    relevant_gene_frequencies,
    relevant_gene_correlation,
    heritabilities,
    heritability_correlation,
    "Test1",
)
