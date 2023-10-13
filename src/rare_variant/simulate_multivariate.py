import numpy as np
import pickle
import scipy.stats as st

from multivariate_bernoulli import multivariate_bernoulli_rvs


def simulate_rare_variate_multivariate(
    prevalences_disorders,
    population_size,
    number_of_genes,
    mutation_rate,
    relevant_gene_frequencies,
    relevant_gene_correlation,
    heritabilities,
    heritability_correlation,
    save_name,
    seed=2021,
):
    """
    This simulates from many conditions. Since these are rare diseases we 
    are going to pretend that everybody has just one copy

    Parameters
    ----------
    prevalences_disorders : array-like,(n_phenotypes,)
        The prevalences of each of the phenotypes observed

    population_size : int
        The number of people on earth

    number_of_genes : int
        The number of genes in our population
    
    mutation_rate : float
        The genetic mutation rate shared across all genes/people

    relevant_gene_frequencies : array-like,(n_phenotypes,)
        The frequencies that genes relate to a specific phenotype

    relevant_gene_correlation : array-like,(n_phenotypes,n_phenotypes)
        We now assume that there are correlations in which genes are 
        relevant to multiple phenotypes. Avoids specifying the probs

    heritabilities : array-like,(n_phenotypes,)

    heritabilities: array-like,(n_phenotypes,n_phenotypes)

    save_name : str
        The name of the pickle file

    seed : int,default=2021
        The random number generation seed

    """
    n_phenotypes = len(heritabilities)
    rng = np.random.default_rng(seed)
    output_dictionary = {}
    output_dictionary["prevalences_disorders"] = prevalences_disorders
    output_dictionary["population_size"] = population_size
    output_dictionary["number_of_genes"] = number_of_genes
    output_dictionary["mutation_rate"] = mutation_rate
    output_dictionary["relevant_gene_frequencies"] = relevant_gene_frequencies
    output_dictionary["relevant_gene_correlation"] = relevant_gene_correlation
    output_dictionary["heritabilities"] = heritabilities
    output_dictionary["heritability_correlation"] = heritability_correlation
    output_dictionary["seed"] = seed

    thresholds = st.norm.ppf(1 - prevalences_disorders)
    output_dictionary["thresholds"] = thresholds

    expected_mutations = population_size * mutation_rate

    # First step is to figure out which genes are affected, we already have
    # the means of genes being affected, need to extract the covariance
    # matrix.

    # Figure out which genes cause problems. Produces a n_genes x n_phenotypes
    gene_architecture = multivariate_bernoulli_rvs(relevant_gene_frequencies,
        relevant_gene_correlation, seed, n_samples=number_of_genes
    )
    output_dictionary["gene_architecture"] = gene_architecture

    # Go through and figure out who has stuff wrong with them.
    # n_people x n_genes
    rare_allele_carriers = np.zeros((population_size, number_of_genes))
    for i in range(number_of_genes):
        variant_count = rng.poisson(expected_mutations)
        idx_affected = rng.choice(
            population_size, size=variant_count, replace=False
        )
        rare_allele_carriers[idx_affected, i] = 1
    output_dictionary["rare_allele_carriers"] = rare_allele_carriers

    # We keep heritabilities as before, but now we use the actual
    # occurance rather than just the estimated probabilities
    #
    sigma = np.sqrt(
        heritabilities
        / (
            mutation_rate
            * (1 - mutation_rate)
            * np.sum(gene_architecture, axis=0)
        )
    )
    mu = np.zeros(n_phenotypes)
    covariance_her = np.zeros((n_phenotypes, n_phenotypes))
    for i in range(n_phenotypes):
        for j in range(n_phenotypes):
            covariance_her[i, j] = (
                heritability_correlation[i, j] * sigma[i] * sigma[j]
            )

    # We are going to figure out the liabliity of every individual
    liability_expected = np.zeros((population_size, n_phenotypes))
    liability = np.zeros((population_size, n_phenotypes))

    total_variance = np.zeros(n_phenotypes)

    # Now we need to compute the liabilities of everyone.
    for i in range(number_of_genes):
        # Only continue if gene impacts any phenotype and people have it
        gene_arch = gene_architecture[i]
        p_variant = np.sum(rare_allele_carriers[:, i]) / population_size
        if (np.sum(gene_arch) > 0) & (p_variant > 0):

            cov_sub = covariance_her[np.ix_(gene_arch == 1, gene_arch == 1)]
            mu_sub = mu[gene_arch == 1]
            beta = np.abs(st.multivariate_normal.rvs(mu_sub, cov_sub))

            variances = 2 * p_variant * (1 - p_variant) * beta ** 2
            total_variance[gene_arch == 1] += variances

            alpha0 = -beta * p_variant
            # Subtract off a bit of liability from everyone
            liability_expected[:, gene_arch == 1] += alpha0
            # Add a lot for those with the variant
            a = gene_arch==1
            b = rare_allele_carriers[:,i] == 1
            liability_expected[np.ix_(b, a)] += beta

    # Now we need to go through and normalize
    genetic_liability_mean = np.mean(liability_expected, axis=0)
    genetic_liability_var = np.var(liability_expected, axis=0)
    res_liability_mean = np.zeros(n_phenotypes)
    res_liability_var = np.zeros(n_phenotypes)

    affected = np.zeros((population_size, n_phenotypes))

    residual_variance = 1.0 - heritabilities
    liability = liability_expected + st.multivariate_normal.rvs(
        np.zeros(n_phenotypes),
        np.diag(residual_variance ** 2),
        size=population_size,
    )

    affected = 1.0 * (liability > thresholds)

    res_liability_mean = res_liability_mean / population_size
    res_liability_var = res_liability_var / population_size
    genetic_liability_var -= genetic_liability_mean ** 2
    res_liability_var -= res_liability_mean ** 2

    output_dictionary["total_affected"] = np.sum(affected, axis=0)
    output_dictionary["genetic_liability_mean"] = genetic_liability_var
    output_dictionary["genetic_liability_mean"] = genetic_liability_mean
    output_dictionary["res_liability_var"] = res_liability_var
    output_dictionary["res_liability_mean"] = res_liability_mean

    with open(save_name + ".p", "wb") as f:
        pickle.dump(output_dictionary, f)
