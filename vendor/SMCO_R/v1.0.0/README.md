This GitHub repository contains R source code for:
 
Paper: "Optimization via Strategic Law of Large Numbers"

By: Xiaohong Chen, Zengjing Chen, Wayne Yuan Gao, Xiaodong Yan, and Guodong Zhang 

Date: July 16, 2025

GitHub Repository Maintained by: Wayne Yuan Gao

For quesions, comments, and bug reports, contact: waynegao@upenn.edu


--------------------------------------------------------------------------
Core Rscripts:

(1) SMCO.R: contains the main Strategic Monte Carlo Optimization (SMCO) algorithm

(2) RunComparison.R: runs comparisons of SMCO with other optimization algorithms, as described and reported in the Numerical Experiments section of the Paper above.

Other Supporting Rscripts:

(3) testfuncs.R: encodes various deterministic test functions and creates test configurations for RunComparison.R

(4) testfunc_NN1L.R: encodes random test functions based on neural networks and create test configurations for RunComparison.R

(5) GD.R: implements Gradient Descent

(6) SignGD.R: implements Sign Gradient Descent

(7) SPSA.R: implements Simultaenous Perturbation Stochastic Approximation

--------------------------------------------------------------------------
User's Guide:

(a) SMCO.R is self-contained and can be used as a general-purpose optimizer

(b) To replicate the Numerical Experiments in the Paper:
	(b.i) Put all the Rscripts in the same directory.
	(b.ii) Configure the last section of RunComparison.R to select test configurations and optimization algorithms to include.
	(b.iii) Execute RunComparison.R
