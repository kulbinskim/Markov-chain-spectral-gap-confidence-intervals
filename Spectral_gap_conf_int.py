import numpy as np
import seaborn as sns
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from scipy.optimize import root
from scipy.linalg import eig

def mc_simulator(initial_distr, transition_matrix, sample_length):
  """
  simulates Markov chain trajectory

  Parameters:
  ----------
  initial_distr: array
    initial distribution of a chain
  transition_matrix: array
    transition matrix of a chain
  smaple_length: int
    desired length of trajectory

  Returns
  ------
  array
    Array contains a simulated path of desired length
  """
  trajectory = np.zeros(sample_length)
  current_distr = np.copy(initial_distr)
  for i in range(sample_length):
    l = np.random.multinomial(1, current_distr)
    current_state = np.where(l == 1)[0].item()
    trajectory[i] =current_state
    current_distr = transition_matrix[current_state, :]
  return trajectory



def conf_int_tester(sample_path, how_many_states, alpha, transition_matrix):
  """
  Calculates spectral gap estimator with confidence intervals
  from a single trajectory of a Markov chain.

  Parameters:
  ----------
  sample_path: array
    trajectory of a chain
  how_many_states: int
    Number of states of a chain from which a sample trajectory was obtained
  alpha: float
    confidence level for confidence intervals for spectral gap estimator

  Returns:
  -------

  """
  sample_size = sample_path.size
  N_vector = np.zeros(how_many_states)
  N_matrix = np.zeros((how_many_states, how_many_states))
  for iter in range(sample_size - 1):
    i, j = int(sample_path[iter]), int(sample_path[iter + 1])
    N_matrix[i, j] += 1
    N_vector[i] += 1
  P_hat = np.zeros((how_many_states, how_many_states))
  for i in range(how_many_states):
    for j in range(how_many_states):
      P_hat[i,j] = (N_matrix[i,j] + 1/how_many_states)/(N_vector[i] + 1)

  # Generating group inverse
  A_hat = np.identity(how_many_states) - P_hat
  A_hat_hash = np.linalg.pinv(A_hat)
  # Estimator for stationary distribution generated from P_hat
  pi_hat = stationary_distribution(P_hat)
  pi_hat = np.real_if_close(pi_hat, tol=1e5) # we add this part to avoid numerical errors

  # Generating estimator for spectral gap

  L_hat = (np.diag(pi_hat)**(1/2)) @ P_hat @ np.diag(pi_hat**(-1/2))
  sym_L_hat = (L_hat + L_hat.T)/2

  eigenvalues_L_hat, eigenvectors_L_hat = np.linalg.eig(sym_L_hat)

  eigenvalues_L_hat = np.sort(eigenvalues_L_hat)[::-1]

  spectral_gap_hat = 1 - np.max(np.array([eigenvalues_L_hat[1], np.abs(eigenvalues_L_hat[how_many_states-1])]))


  c = 1.1


  minimal_t = root_scalar(aux_tau_function,args =(c,how_many_states, sample_size, alpha),x0 = 5).root
  B_hat = (np.sqrt((c*minimal_t)/(2*N_vector)) + np.sqrt((c*minimal_t)/(2*N_vector) +
                                                   np.sqrt(2*c*P_hat*(1 - P_hat)*minimal_t)/N_vector) +
                                                    ((4/3)*minimal_t + np.abs(P_hat - 1/how_many_states))/N_vector)**2

  minima = np.min(A_hat_hash, axis = 0)
  max = np.max(np.diagonal(A_hat_hash) - minima)
  condition_number_hat = max/2
  bound_for_pi = condition_number_hat*np.max(B_hat)

  # Empirical bounds for | sepctral_gap - spectral_hap_hat |
  arr1 = bound_for_pi/pi_hat
  test_zeros = np.reshape(np.zeros(how_many_states), (1, how_many_states))
  test_vals = np.reshape(pi_hat - bound_for_pi, (1, how_many_states))
  test_arr = np.concatenate((test_vals,test_zeros), axis = 0)
  maxes = np.max(test_arr, axis=0)
  arr2 = bound_for_pi/maxes
  test_arr_2 = np.concatenate((arr1, arr2))
  ro_hat = (1/2)*np.max(test_arr_2)
  B_aux = np.sqrt(np.diag(pi_hat)) @ B_hat @ np.sqrt(np.diag(1/pi_hat))
  bound_for_spectral_gap = 2*ro_hat + ro_hat**2 + (1 + 2*ro_hat + ro_hat**2)*(np.sum(B_aux**2))**(1/2)
  real_pi = stationary_distribution(transition_matrix)
  max_diff = np.max(np.abs(pi_hat - real_pi))
  print("real spectral gap:", spectral_gap(transition_matrix), "\nSpectral gap estimator:", spectral_gap_hat, "\nSpectral gap bound: ", bound_for_spectral_gap,
        "\nSpectral gap abs diffrence: ", np.abs(spectral_gap(transition_matrix) - spectral_gap_hat),
        "\nSpectral gap conf int: ", np.array([spectral_gap_hat- bound_for_spectral_gap, spectral_gap_hat + bound_for_spectral_gap]),
        "\nSpectral gap conf int length: ", bound_for_spectral_gap*2,
        "\nStationary distribution estimator: ", pi_hat, "\nMaximal diffrence between estimated distribution and estimator: ", max_diff, "\nStationary distr bound: ", bound_for_pi)



def mc_distribution_calculator(initial_distr, transition_matrix, how_many_iters):
  """
  Calculates distributions of markov chain in each of desired number of iterations.

  Parameters:
  ---------
  initial_distr:array
    initial distribution of a chain
  transition_matrix: array
    transition matrix of a chain
  how_many_inters: int
    desired number of iterations

  Returns
  ------
  arrray
    In k-th row array contains distribution of a chain in k-th step
  """

  current_distr = np.copy(initial_distr)
  distr_history = np.zeros((how_many_iters, np.size(initial_distr)))
  for i in range(how_many_iters):
    distr_history[i,:] = current_distr
    current_distr = current_distr @ transition_matrix
  return distr_history


def walk_on_graph_transition_matrix(adj_matrix):
  """
  Provides a transition matrix of Markov chain walking od undirected graph.
  Each adjacent node is chosen with equal probability.

  Parameters:
  ---------
  adj_matrix: array
    Adjacency matrix of a graph containing zeros and ones.
    If a desired graph has n edges than is should be nxn array.
    Ones ancode a vertex between nodes.
    Zeros ancode no vertex between nodes.

  Returns:
  -------
  array
    Transition matrix of a chain walking on desired graph

  """

  how_many_states = np.shape(adj_matrix)[0]
  transition_matrix = np.zeros((how_many_states, how_many_states))
  for row in range(how_many_states):
    edges_idx = np.argwhere(adj_matrix[row] == 1)
    how_many_edges = np.size(edges_idx)
    probability = 1/how_many_edges
    for i in range(how_many_edges):
      transition_matrix[row, edges_idx[i]] = probability
  return transition_matrix

def birth_and_death_transition_matrix(birth_prob_vec, death_prob_vec, ):
  """
  Provides a transition matrix o a birth and death process.
  Probabilities of staying in the same state are calculated based on desired
  birth and death probabilities.

  Parameters:
  ---------
  birth_prob_vec: array
    vector of length n containing probability of birth occuring in each state
  death_prob_vec: array
    vector of length n containing probability of death occuring in each state

  Returns
  ------
  array
    Transition matrix of a chain with desired probabilities of death and birth

  """
  size = birth_prob_vec.size
  stay_same_pr_vec = np.ones(size) - birth_prob_vec - death_prob_vec
  tr_matrix = np.zeros((size, size))
  np.fill_diagonal(tr_matrix, stay_same_pr_vec)
  for i in range(size-1):
    tr_matrix[i,i+1] = birth_prob_vec[i]
    tr_matrix[i+1,i] = death_prob_vec[i]
  tr_matrix[0,0] = 1 - tr_matrix[0,1]
  tr_matrix[size-1, size-1] = 1 - tr_matrix[size-1, size-2]
  return tr_matrix


def stationary_distribution(transition_matrix):
  """
  Calculates stationary distribution of a Markov Chain.

  Parameters:
  ----------
  transition_matrix: array
    transition matrix of a chain

  Returns
  ------
  int
    stationary distribution of a chain
  """
  eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
  idx = np.argsort(eigenvalues)
  eigenvectors = eigenvectors[:,idx]
  pi = eigenvectors[:,-1] / np.sum(eigenvectors[:,-1])
  return pi

def total_var_dist(distr_1, distr_2):
  """
  calculates total variation distanace between distributions

  Parameters:
  ---------
  distr_1: array
    first distribution
  distr_2: array
    Second distribution. It can be a single distribution or an array containing
    one distribution in each row.

  Returns
  ------
  int or array
    In case of distr_2 being a single distribution function returns
    total variation distance as na integer.
    In case of distr_2 being an array containig a sequence n of distributions
    funtion returns and one-dimensional array of lenght n containing distance between first
    distribution and kth distribution form distr_2 on kth place.
  """
  # distribution 1 has to be one-dimensional
  # distribution 2 can be two dimensional array
  ax = distr_2.ndim - 1
  tv_norm = np.sum(np.abs(distr_1 - distr_2), axis = ax)/2
  return tv_norm

def mixing_time(tr_matrix, epsilon):
  """
  Calculates mixing time for a Markov chain with given transition matrix
  and desired epsilon

  Parameters:
  ----------
  tr_matrix: array
    transition matrix of a chain for which mixing time has to be calculated
  epsilon: float
    positive number

  Returns:
  -------
  int
    mixing time for given transition matrix and epsilon value
  """

  ## finds minimal number of steps t such that maximal_dist(tr_matrix, step) < epsilon
  max_distance = float('inf')
  t = 1
  stationary = stationary_distribution(tr_matrix)
  current_matrix = np.copy(tr_matrix)
  while max_distance > epsilon:
    max_distance = np.max(total_var_dist(stationary, current_matrix))
    current_matrix = current_matrix @ tr_matrix
    t += 1
  return t

def spectral_gap(transition_matrix):
  """
  Calculates spectral gap of Markov Chain with given transition matrix

  Parameters:
  ----------
  transition_matrix: array
    transition matrix of a markov chain

  Returns:
  -------
  int
    spectral gap caluclated form given transition matrix
  """
  eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
  test_array = np.ones(eigenvalues.size)
  idx_to_cancell = np.where(np.isclose(eigenvalues,test_array))[0]
  eigenvalues = np.delete(eigenvalues, idx_to_cancell)
  spectral_gap = 1 - np.max(np.abs(eigenvalues))
  return spectral_gap

def spectral_gap_and_eig(transition_matrix):
  eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
  test_array = np.ones(eigenvalues.size)
  idx_to_cancell = np.where(np.isclose(eigenvalues,test_array))[0]
  eigenvalues = np.delete(eigenvalues, idx_to_cancell)
  spectral_gap = 1 - np.max(np.abs(eigenvalues))
  return spectral_gap, np.sort(eigenvalues)



def bounds_for_mixing(tr_matrix, epsilon):
  """
  Calculatex bounds for mixing time described in Markov Chains and Mixing Times by
  Levin and Peres.
  Bounds are calculated based on transition matrix and epsilon value

  Parameters:
  ----------
  tr_matris: array
    transition matrix of a chain
  epsilon: float
    positive number

  Returns:
  -------
  two integers
    First integer is a lower bound for mixing time
    Second integer is an upper bound for mixing time
  """
  t_relax = 1/spectral_gap(tr_matrix)
  stationary = stationary_distribution(tr_matrix)
  pi_min = np.min(stationary)
  upper_bound = t_relax*math.log(1/(epsilon*pi_min))
  lower_bound = (t_relax - 1)*math.log(1/(2*epsilon))
  return lower_bound, upper_bound

def birth_death(chain_size, birth_pr, death_pr):
  """
    This is a simplified version of the function birth_and_death_transition_matrix.
    It creates transition matix with the same probabilities of birth and death
    in each state.

    Parameters:
    ----------
    chain_size: int
      number of states in a chain
    birth_pr: float
      probability of birth occuring
    death_pr: float
      probability of death occurign

    Returns:
    --------
    array
      transition matrix of a chain with demanded birth and death probabilities

  """
  birth_vec = np.zeros(chain_size)
  birth_vec.fill(birth_pr)
  death_vec = np.zeros(chain_size)
  death_vec.fill(death_pr)
  stay_vec = np.ones(chain_size-1) - birth_vec - death_vec
  birth_vec[0] = 1 - stay_vec[0]
  death_vec[chain_size-2] = 1 - stay_vec[chain_size-1]
  tr_matrix = birth_and_death_transition_matrix(birth_vec, death_vec)
  return tr_matrix

def convergence_tracker(tr_matrix, steps, initial_distr):
  distributions =  mc_distribution_calculator(initial_distr, tr_matrix, steps)
  stationaty_distr = stationary_distribution(tr_matrix)
  history = np.zeros(steps)
  for i in range(steps):
    history[i] = total_var_dist(stationaty_distr, distributions[i])
  plt.plot(history)
  return history

def positive_ceiling_for_arrays(x):
  """
  Calculates maximum between ceiling function of n array x and zero
  """
  new_x = np.concatenate((np.array([0]), x))
  return np.max(np.ceil(new_x))


def positive_ceiling(x):
  """
  Calculates maximum between ceiling function of a number x and zero
  """
  if isinstance(x, np.ndarray):
        x = x.item()  # Convert single-element arrays to scalars
  return max(0, math.ceil(x))

def aux_tau_function(t,c,d,n,delta):
  """
  Calculates a value of an auxiliary function needed to obtain a tau value.
  Tau value is used in algorithm computing empirical confidence intervals for
  spectral gap of a Markov Chain

  Parameters:
  ----------
  t: float
    an agrument of a function
  c: float
    additional constant,for example c = 1.1
  d: int
    number of states in a chain
  n: int
    lenght of a path used to calulate confidence intervals
  delta: float
    confidence level for a confidence intervals for spectral gap.

  Returns:
  -------
  float
    value of an ausiliary function for given t and parameters

  """
  if isinstance(t, np.ndarray):
        t = t.item()
  return delta - 2*(d**2)*(1 + positive_ceiling(math.log((2*n)/t, c)))*math.exp(-t)


def main_fun(sample_path,how_many_states,alpha):
  """
  Calculates spectral gap estimator and stationary distribution estimator
   with confidence intervals from a single trajectory of a Markov chain.

  Parameters:
  ----------
  sample_path: array
    sample trajectory of a chain
  how_many_states: int
    Number of states of a chain from which the sample trajectory was obtained
  alpha: float
    confidence level for confidence intervals for spectral gap estimator and
    stationary distribution estimator

  Returns:
  -------
  spectral_gap_hat: float
    Estimator of a spectral gap calculated based on sample path
  confidence_interval: array
    An one dimensional array of lenght 2 containing lower and upper bound of
    confidence interval for a spectral gap with confidence level alpha
  confidence_interval_length: float
    a length of confidence interval for spectral gap
  pi_hat: array
    Estimator of stationary distribution of a chain
  bound_for_pi: float
    A bound for an max of absolute values between estimated and real stationary
    probabilities

  Description:
  -----------
  This function calculates  estimators of spectral gap and stationary distribution
   with fully empirical (meaning based on a given trajectory only) confidence intervals on given
  confidence level alpha.
  The procedure used in this function comes from paper Mixing Time Estimation
  in Reversible Markov Chains from a Single Sample Path by Hsu, Kontorovich
  and  Szepesv´ari.

  """

  sample_size = sample_path.size
  N_vector = np.zeros(how_many_states)
  N_matrix = np.zeros((how_many_states, how_many_states))
  for iter in range(sample_size - 1):
    i, j = int(sample_path[iter]), int(sample_path[iter + 1])
    N_matrix[i, j] += 1
    N_vector[i] += 1
  P_hat = np.zeros((how_many_states, how_many_states))
  for i in range(how_many_states):
    for j in range(how_many_states):
      P_hat[i,j] = (N_matrix[i,j] + 1/how_many_states)/(N_vector[i] + 1)

  # Generating group inverse
  A_hat = np.identity(how_many_states) - P_hat
  A_hat_hash = np.linalg.pinv(A_hat)
  # Estimator for stationary distribution generated from P_hat
  pi_hat = stationary_distribution(P_hat)
  pi_hat = np.real_if_close(pi_hat, tol=1e5)

  # Generating estimator for spectral gap

  L_hat = (np.diag(pi_hat)**(1/2)) @ P_hat @ np.diag(pi_hat**(-1/2))
  sym_L_hat = (L_hat + L_hat.T)/2

  eigenvalues_L_hat, eigenvectors_L_hat = np.linalg.eig(sym_L_hat)

  eigenvalues_L_hat = np.sort(eigenvalues_L_hat)[::-1]

  spectral_gap_hat = 1 - np.max(np.array([eigenvalues_L_hat[1], np.abs(eigenvalues_L_hat[how_many_states-1])]))
  c = 1.1


  minimal_t = root_scalar(aux_tau_function,args =(c,how_many_states, sample_size, alpha),x0 = 5).root
  B_hat = (np.sqrt((c*minimal_t)/(2*N_vector)) + np.sqrt((c*minimal_t)/(2*N_vector) +
                                                   np.sqrt(2*c*P_hat*(1 - P_hat)*minimal_t)/N_vector) +
                                                    ((4/3)*minimal_t + np.abs(P_hat - 1/how_many_states))/N_vector)**2

  minima = np.min(A_hat_hash, axis = 0)
  max = np.max(np.diagonal(A_hat_hash) - minima)
  condition_number_hat = max/2
  bound_for_pi = condition_number_hat*np.max(B_hat)
  confidence_interval_for_pi = np.array([[pi_hat - bound_for_pi], [pi_hat + bound_for_pi]])
  # Empirical bounds for | sepctral_gap - spectral_hap_hat |
  arr1 = bound_for_pi/pi_hat
  test_zeros = np.reshape(np.zeros(how_many_states), (1, how_many_states))
  test_vals = np.reshape(pi_hat - bound_for_pi, (1, how_many_states))
  test_arr = np.concatenate((test_vals,test_zeros), axis = 0)
  maxes = np.max(test_arr, axis=0)
  arr2 = bound_for_pi/maxes
  test_arr_2 = np.concatenate((arr1, arr2))
  ro_hat = (1/2)*np.max(test_arr_2)
  B_aux = np.sqrt(np.diag(pi_hat)) @ B_hat @ np.sqrt(np.diag(1/pi_hat))
  bound_for_spectral_gap = 2*ro_hat + ro_hat**2 + (1 + 2*ro_hat + ro_hat**2)*(np.sum(B_aux**2))**(1/2)
  confidence_interval = np.array([spectral_gap_hat- bound_for_spectral_gap, spectral_gap_hat + bound_for_spectral_gap])
  confidence_interval_length = bound_for_spectral_gap*2

  return spectral_gap_hat,confidence_interval, confidence_interval_length, pi_hat, confidence_interval_for_pi, bound_for_pi

  

def conf_int_tester(sample_path, how_many_states, alpha, transition_matrix):
  """
  Calculates spectral gap estimator with confidence intervals
  from a single trajectory of a Markov chain.

  Parameters:
  ----------
  sample_path: array
    trajectory of a chain
  how_many_states: int
    Number of states of a chain from which a sample trajectory was obtained
  alpha: float
    confidence level for confidence intervals for spectral gap estimator

  Returns:
  -------

  """
  sample_size = sample_path.size
  N_vector = np.zeros(how_many_states)
  N_matrix = np.zeros((how_many_states, how_many_states))
  for iter in range(sample_size - 1):
    i, j = int(sample_path[iter]), int(sample_path[iter + 1])
    N_matrix[i, j] += 1
    N_vector[i] += 1
  P_hat = np.zeros((how_many_states, how_many_states))
  for i in range(how_many_states):
    for j in range(how_many_states):
      P_hat[i,j] = (N_matrix[i,j] + 1/how_many_states)/(N_vector[i] + 1)

  # Generating group inverse
  A_hat = np.identity(how_many_states) - P_hat
  A_hat_hash = np.linalg.pinv(A_hat)
  # Estimator for stationary distribution generated from P_hat
  pi_hat = stationary_distribution(P_hat)
  pi_hat = np.real_if_close(pi_hat, tol=1e5) # we add this part to avoid numerical errors

  # Generating estimator for spectral gap

  L_hat = (np.diag(pi_hat)**(1/2)) @ P_hat @ np.diag(pi_hat**(-1/2))
  sym_L_hat = (L_hat + L_hat.T)/2

  eigenvalues_L_hat, eigenvectors_L_hat = np.linalg.eig(sym_L_hat)

  eigenvalues_L_hat = np.sort(eigenvalues_L_hat)[::-1]

  spectral_gap_hat = 1 - np.max(np.array([eigenvalues_L_hat[1], np.abs(eigenvalues_L_hat[how_many_states-1])]))


  c = 1.1


  minimal_t = root_scalar(aux_tau_function,args =(c,how_many_states, sample_size, alpha),x0 = 5).root
  B_hat = (np.sqrt((c*minimal_t)/(2*N_vector)) + np.sqrt((c*minimal_t)/(2*N_vector) +
                                                   np.sqrt(2*c*P_hat*(1 - P_hat)*minimal_t)/N_vector) +
                                                    ((4/3)*minimal_t + np.abs(P_hat - 1/how_many_states))/N_vector)**2

  minima = np.min(A_hat_hash, axis = 0)
  max = np.max(np.diagonal(A_hat_hash) - minima)
  condition_number_hat = max/2
  bound_for_pi = condition_number_hat*np.max(B_hat)

  # Empirical bounds for | sepctral_gap - spectral_hap_hat |
  arr1 = bound_for_pi/pi_hat
  test_zeros = np.reshape(np.zeros(how_many_states), (1, how_many_states))
  test_vals = np.reshape(pi_hat - bound_for_pi, (1, how_many_states))
  test_arr = np.concatenate((test_vals,test_zeros), axis = 0)
  maxes = np.max(test_arr, axis=0)
  arr2 = bound_for_pi/maxes
  test_arr_2 = np.concatenate((arr1, arr2))
  ro_hat = (1/2)*np.max(test_arr_2)
  B_aux = np.sqrt(np.diag(pi_hat)) @ B_hat @ np.sqrt(np.diag(1/pi_hat))
  bound_for_spectral_gap = 2*ro_hat + ro_hat**2 + (1 + 2*ro_hat + ro_hat**2)*(np.sum(B_aux**2))**(1/2)
  real_pi = stationary_distribution(transition_matrix)
  max_diff = np.max(np.abs(pi_hat - real_pi))
  print("real spectral gap:", spectral_gap(transition_matrix), "\nSpectral gap estimator:", spectral_gap_hat, "\nSpectral gap bound: ", bound_for_spectral_gap,
        "\nSpectral gap abs diffrence: ", np.abs(spectral_gap(transition_matrix) - spectral_gap_hat),
        "\nSpectral gap conf int: ", np.array([spectral_gap_hat- bound_for_spectral_gap, spectral_gap_hat + bound_for_spectral_gap]),
        "\nSpectral gap conf int length: ", bound_for_spectral_gap*2,
        "\nStationary distribution estimator: ", pi_hat, "\nMaximal diffrence between estimated distribution and estimator: ", max_diff, "\nStationary distr bound: ", bound_for_pi)
