import numpy as np
import numpy.random as rand
import pickle
import scipy.stats as st
import sys

def main(
    prevalence_disorder1,
    prevalence_disorder2,
    sample_size,
    number_of_genes,
    mean_rare_freq_per_gene,
    fraction_gene1,
    fraction_gene2,
    fraction_both,
    rare_h2_1,
    rare_h2_2,
    rho,
    outname,
):
    output_dictionary = {}
    output_dictionary["prevalence_disorder1"] = prevalence_disorder1
    output_dictionary["prevalence_disorder2"] = prevalence_disorder2
    output_dictionary["sample_size"] = sample_size
    output_dictionary["number_of_genes"] = number_of_genes
    output_dictionary["mean_rare_freq_per_gene"] = mean_rare_freq_per_gene
    output_dictionary["fraction_gene1"] = fraction_gene1
    output_dictionary["fraction_gene2"] = fraction_gene2
    output_dictionary["rare_h2_1"] = rare_h2_1
    output_dictionary["rare_h2_2"] = rare_h2_2
    output_dictionary["rho"] = rho

    rand.seed(2021)
    thresholds = np.zeros(2)
    thresholds[0] = st.norm.ppf(1 - prevalence_disorder1)
    thresholds[1] = st.norm.ppf(1 - prevalence_disorder2)
    output_dictionary["thresholds"] = thresholds

    probs_diseases = np.zeros(4)
    probs_diseases[0] = 1 - fraction_gene1 - fraction_gene2 - fraction_both
    probs_diseases[1] = fraction_gene1
    probs_diseases[2] = fraction_gene2
    probs_diseases[3] = fraction_both

    p_disease = np.zeros(2)
    p_disease[0] = fraction_gene1
    p_disease[1] = fraction_gene2

    heritability = np.zeros(2)
    heritability[0] = rare_h2_1
    heritability[1] = rare_h2_2

    lamb = 2 * sample_size * mean_rare_freq_per_gene

    sigma = np.sqrt(
        heritability
        / (
            2
            * mean_rare_freq_per_gene
            * (1 - mean_rare_freq_per_gene)
            * (p_disease + fraction_both)
            * number_of_genes
        )
    )

    mu = np.zeros(2)
    expected_mutations_individual = lamb * number_of_genes
    mutations_affecting_disease = expected_mutations_individual * (
        1 - probs_diseases[0]
    )

    output_dictionary['mutations_affecting_disease'] = mutations_affecting_disease

    gene_architectures = -1 * np.ones(number_of_genes)
    rare_allele_carriers = np.zeros(
        (sample_size, number_of_genes)
    )  # Matrix instead of ragged array
    liability = np.zeros((sample_size, 2))
    cheat = np.zeros((number_of_genes, 6))

    total_variance = np.zeros(2)
    stupid_sum = np.zeros(2)

    for i in range(number_of_genes):
        variant_count = rand.poisson(lamb)

        this_stupid = np.zeros(2)
        if variant_count == 0:
            gene_architectures[i] = -1
            cheat[i] = np.array([i, -1, 0, 0, 0, 0])
        if variant_count > 0:
            # Assign individuals to have variant
            idx_affected = rand.choice(
                sample_size, size=variant_count, replace=False
            )
            rare_allele_carriers[idx_affected, i] = 1

            # Assign gene to have disease or not
            disease_status = rand.choice(4, p=probs_diseases)
            gene_architectures[i] = disease_status

            if disease_status == 3:  # Gene affects both diseses
                cheat[i] = np.array([i, 3, rare_h2_1, rho, rho, rare_h2_2])
                p_disease = variant_count / (2 * sample_size)
                q_disease = 1 - p_disease

                cov = np.zeros((2, 2))
                cov[0, 0] = sigma[0] ** 2
                cov[1, 1] = sigma[1] ** 2
                cov[0, 1] = sigma[0] * sigma[1] * rho
                cov[1, 0] = cov[0, 1]

                beta = st.multivariate_normal.rvs(mu, cov)
                if beta[0] < 0:
                    if beta[1] < 0:
                        beta = -1 * beta
                    elif np.abs(beta[0]) > np.abs(beta[1]):
                        beta = -1 * beta
                elif beta[1] < 0:
                    if np.abs(beta[1]) > np.abs(beta[0]):
                        beta = -1 * beta

                #alpha0 = beta[0] * np.array([-p_disease, q_disease])
                alpha0 = - beta * p_disease

                variances = 2 * p_disease * q_disease * beta ** 2
                total_variance += variances
                a2 = 2 * alpha0
                for indiv in range(sample_size):
                    liability[indiv] += a2
                    stupid_sum += a2
                    this_stupid += a2

                liability[rare_allele_carriers[:, i] == 1] += beta
                stupid_sum = stupid_sum + (variant_count * beta)
                this_stupid = this_stupid + (variant_count * beta)
            elif disease_status == 1:  # Gene affects disease 1
                cheat[i] = np.array([i, 1, rare_h2_1, 0, 0, 0])
                p = variant_count / (2 * sample_size)
                q = 1 - p
                beta = np.abs(rand.randn(1) * sigma[0] + mu[0]) # abs(N(mu[0],sigma[0]^2))
                alpha0 = -p * beta
                variance = 2 * p * q * beta ** 2
                a2 = 2 * alpha0
                total_variance[0] += variance

                liability[:, 0] += a2
                stupid_sum[0] += a2*sample_size
                this_stupid[0] += a2*sample_size

                liability[rare_allele_carriers[:, i] == 1, 0] += beta
                stupid_sum[0] += (variant_count * beta)
                this_stupid[0] += (variant_count * beta)

            elif disease_status == 2:  # Gene affects disease 2
                cheat[i] = np.array([i, 2, 0, 0, 0, rare_h2_2])
                p = variant_count / (2 * sample_size)
                q = 1 - p
                beta = np.abs(rand.randn(1) * sigma[1] + mu[1])
                alpha0 = -p * beta
                variance = 2 * p * q * beta ** 2
                a2 = 2 * alpha0
                total_variance[1] += variance

                liability[:, 1] += a2
                stupid_sum[1] += a2*sample_size
                this_stupid[1] += a2*sample_size

                liability[rare_allele_carriers[:, i] == 1, 1] += beta
                stupid_sum[1] += (variant_count * beta)
                this_stupid[1] += (variant_count * beta)

            else:  # Gene affects neither
                cheat[i] = np.array([i, 0, 0, 0, 0, 0])

    # Now go through and normalize things
    output_dictionary["stupid_sum"] = stupid_sum

    genetic_liability_mean = np.mean(liability, axis=0)
    genetic_liability_var = np.mean(liability ** 2, axis=0)
    res_liability_mean = np.zeros(2)
    res_liability_var = np.zeros(2)

    affected = np.zeros((sample_size, 2))

    res_var = 1.0 - heritability

    for i in range(sample_size):
        this_e = rand.randn(2)*res_var
        res_liability_mean += this_e
        res_liability_var += this_e ** 2
        liability[i] += this_e
        if liability[i, 0] >= thresholds[0]:
            affected[i, 0] = 1
        if liability[i, 1] >= thresholds[1]:
            affected[i, 1] = 1

    res_liability_mean = res_liability_mean / sample_size
    res_liability_var = res_liability_var / sample_size
    genetic_liability_var -= genetic_liability_mean ** 2
    res_liability_var -= res_liability_mean ** 2

    output_dictionary["total_affected"] = np.sum(affected, axis=0)
    output_dictionary["genetic_liability_mean"] = genetic_liability_var
    output_dictionary["genetic_liability_mean"] = genetic_liability_mean
    output_dictionary["res_liability_var"] = res_liability_var
    output_dictionary["res_liability_mean"] = res_liability_mean

    np.savetxt(
        outname + "_carriers.csv", rare_allele_carriers, fmt="%d", delimiter=","
    )
    np.savetxt(outname + "_liability.csv", liability, fmt="%d", delimiter=",")
    np.savetxt(outname + "_cheat.csv", cheat, fmt="%d", delimiter=",")
    np.savetxt(outname + "_affected.csv", affected, fmt="%d", delimiter=",")
    np.savetxt(outname + "_architectures", gene_architectures, fmt="%d", delimiter=",")
    with open(outname + ".p", "wb") as f:
        pickle.dump(output_dictionary, f)


if __name__ == "__main__":
    prevalence_disorder1 = float(sys.argv[1])
    prevalence_disorder2 = float(sys.argv[2])
    sample_size = int(sys.argv[3])
    number_of_genes = int(sys.argv[4])
    mean_rare_freq_per_gene = float(sys.argv[5])
    fraction_gene1 = float(sys.argv[6])
    fraction_gene2 = float(sys.argv[7])
    fraction_both = float(sys.argv[8])
    rare_h2_1 = float(sys.argv[9])
    rare_h2_2 = float(sys.argv[10])
    rho = float(sys.argv[11])
    outname = sys.argv[12]
    main(
        prevalence_disorder1,
        prevalence_disorder2,
        sample_size,
        number_of_genes,
        mean_rare_freq_per_gene,
        fraction_gene1,
        fraction_gene2,
        fraction_both,
        rare_h2_1,
        rare_h2_2,
        rho,
        outname,
    )
