# Markov-chain-spectral-gap-confidence-intervals



This is my implementation of an algorithm proposed in the paper Mixing Time Estimation in Reversible Markov Chains from a Single Sample Path by Daniel Hsu, Aryeh Kontorovich, and Csaba Szepesvári. You can check out the paper on arXiv via this [link](https://arxiv.org/abs/1506.02903). The implementation was originally a part of my master's thesis in the Math and CS Department at the University of Wrocław.

## Description of the problem

Markov chains appear in various contexts within the fields of machine learning and data science, including applications in reinforcement learning and Markov Chain Monte Carlo (MCMC) sampling. One important problem in Markov chain theory is investigating the rate of convergence to a stationary distribution, usually studied in terms of mixing time. More precisely, investigating the rate of convergence involves finding both lower and upper bounds for the mixing time. There exists a well-established theory for deriving such bounds.

Consider an ergodic and reversible Markov chain ${(X_{0}, X_{1}, X_{2}, \dots)}$ with a finite state space and a transition matrix $P$. Given this setup, one can compute the absolute spectral gap $\gamma_{\star}$ and the stationary distribution $\pi$. Using these quantities, it is possible to establish lower and upper bounds for the mixing time. In this standard formulation of the problem, the transition matrix is assumed to be known.

However, in many real-life scenarios, while we observe a sample path from the chain, we do not have access to its transition matrix. This motivates a modified version of the original problem. Specifically, suppose the only data available is a single trajectory $x_0, x_1, \dots, x_n$ from an ergodic and reversible Markov chain with $d$ states. The question becomes: how can we construct lower and upper bounds for the mixing time given only this data?

A quite natural way of addressing this question is to construct an estimator of the transition matrix $\hat{P}$ based on the frequencies of state transitions observed in a trajectory. Then, using $\hat{P}$, one can obtain an estimator of the stationary distribution $\hat{\pi}$ and the absolute spectral gap $\hat{\gamma}_{\star}$. Finally, these estimated values can be used to derive bounds for the mixing time, similar to the approach taken when the transition matrix is known. However, for these estimators to be meaningful, it is essential to construct confidence intervals around them.

In the paper by Hsu, Kontorovich, and Szepesvári (link at the top of this description), the authors present an algorithm for computing fully data-driven confidence intervals for the estimators of the absolute spectral gap and the stationary distribution of an ergodic and reversible Markov chain with a finite state space. Notably, this approach is the first of its kind.

## About the code


The implementation is done in Python. Together with the algorithm itself, the module contains several utility functions for simulating Markov chains and computing fundamental properties of a given chain.



