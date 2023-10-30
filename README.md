# cwas_simulation
Simulate a connectome wide association study (CWAS) with real data.

This repo contains code used to run a CWAS simulation for the paper "Investigating the convergence of resting-state functional connectivity profiles in Alzheimer’s disease with neuropsychiatric symptoms and schizophrenia".

The CWAS contrasts cases and controls, using whole-brain connectomes with functional connectivity (FC) standardised (z-scored) based on the variance of the control group. CWAS is conducted using linear regression, with z-scored FC as the dependent variable, and case/control label as the explanatory variable. The models are adjusted for site of data collection. The linear regression is applied to each connection (here, 2080), with the resulting β values determining the case FC profile. Number of tests is corrected using Benjamini-Hochberg correction for FDR at a threshold of q < 0.1. The top-10% effect size of the group label on FC is measured as the mean of the top decile of the absolute β values.

The simulation uses real connectomes from neurotypical control participants. The participants are first randomly split into two equal-sized groups. An effect of disease on FC is modelled by altering pi% of connections for one group ("cases"), given by the equation connection<sub>i</sub> = connection<sub>i</sub> + *d* x std, where *d* is equal to a previously published effect size (Cohen’s *d*) for the disease, and std is equal to the standard deviation of the connectivity values combined across groups. This process is repeated (default=100), FDR corrected at a threshold of *q*, and the average sensitivity and specificity calculated.

The notebook `run_simulation.ipynb` runs the simulation to estimate the sensitivity and specificity for a range of sample sizes with a *d* of 0.3, pi of 20% and q of 0.1. Connectomes used in this simulation were derived from the Autism Brain Imaging Data Exchange (ABIDE-1 and -2) initiative.



