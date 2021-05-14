# Scenario layout Parameters ------------------------------------------------------------------------------------------
import numpy as np
instances= 10#10
#Nbs = np.random.randint(4,19)  # 7#3#19
#Nue = np.copy(Nbs)  # 7#3#19
Nbs = 19
Nue = 19

#this parameters are needed only on poisson deployment -----------------------------------------------------------------
width = 180
height = 180
cell_position='grid' # ir cell_position='grid'  OR 'poisson'
# Scenario layout Parameters ------------------------------------------------------------------------------------------

# Neural Network Parameters -------------------------------------------------------------------------------------------
A = 10
intervals = 10
train_intervals = 10
report_interval = 100

warm_ep = 10
num_games = 5000 + warm_ep  # Episodes
exp_games = 3000 + warm_ep
#num_games = 300 + warm_ep  # Episodes
#exp_games = 100 + warm_ep





# ---------------------------------------------------------------------------------------------------------------------

# Validation parameters -----------------------------------------------------------------------------------------------
Validation_Scenarios = 15                               # Number of fixed scenarios
Validate_Freq = 10 # Validation every X episode         # Number of episodes to evaluate the network
AGGREGATE_STATS_EVERY = 50  # episodes
stat_index = int(AGGREGATE_STATS_EVERY/Validate_Freq)   # value for graphs
validation_steps = 5                                    # Number of intervals evaluated on each validation episode
# Validation parameters -----------------------------------------------------------------------------------------------

# Network Parameters --------------------------------------------------------------------------------------------------
Pmax_dBm =23 # 23  # Para este ejercicio (1)
Rcell = 25  # 30
Rmax = 25 #* 0.866
#Rcell = 20
#Rmax = 20

Rmin = 1#
Pmin_dBm = -20
noise_power_dBm = -114  # Para este ejercicio (1)
SINR_th = 10 ** (5 / 10) # Para obtener el SINR en dB -> 10*log10(SNR_th)
# Network Parameters --------------------------------------------------------------------------------------------------

# Initialization ------------------------------------------------------------------------------------------------------

# Neural Network Parameters -------------------------------------------------------------------------------------------

C = 7 # C equals to the number of BS considered on the observation state

chkpt_dir= '/Google Colab/Borrador/Proyecto_Conference/Network_Weights'
