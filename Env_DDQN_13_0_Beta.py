from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import math
import scipy
from scipy import special
from matplotlib import path
from bridson import poisson_disc_samples
import random

''' Differences from Env_DDQN_9_0
Deployment position of UEs on grid deployment corrected
'''

def action_map(action_index, power_avail):
    selected_action = power_avail[action_index]
    return selected_action


def BS_UE_distances(x_sample, y_sample, x_user, y_user):
    distances = np.zeros((len(x_sample), len(x_user)))

    for i in range(len(x_sample)):
        for j in range(len(x_user)):
            distances[i, j] = np.sqrt(((x_user[j] - x_sample[i]) ** 2) + ((y_user[j] - y_sample[i]) ** 2))
    return distances


def association(distances, UEs, BSs):
    counter = np.arange(UEs)
    UE_BS_index = np.ndarray.argmin(distances, 0)
    BS_load = np.zeros(BSs).astype(int)

    while np.count_nonzero(BS_load) != BSs:
        for i in np.arange(BSs):
            if not (len(list(filter(lambda x: x == i, UE_BS_index))) > 0):  # check if item i is on array UE_BS_index
                UE_BS_index[np.ndarray.argmin(distances[i, counter], 0)] = i
                counter = np.delete(counter, np.ndarray.argmin(distances[i, counter], 0))

        BS_load = [np.count_nonzero(UE_BS_index == i) for i in np.arange(BSs)]

    return UE_BS_index, BS_load


def dual_slope_path_loss_matrix(d, K0, alfa1, alfa2, dBP, path_loss):
    path_loss[(d <= dBP)] = K0 + 10 * alfa1 * np.log10(d[(d <= dBP)])
    path_loss[(d > dBP)] = K0 + 10 * alfa2 * np.log10(d[(d > dBP)]) - 10 * (alfa2 - alfa1) * np.log10(dBP)
    return path_loss


def path_loss_LTE(d):
    # d in kilometers d = d/1000
    path_loss = 120.9 + 37.6 * np.log10(d / 1000)
    return path_loss


def shadowing_fading(base_stations, users, dv):
    shadowing = dv * np.random.normal(size=(base_stations, users))
    return shadowing


def scheduling_per_access_point(UE_BS_index, scheduling_counter, channel_gain, active_users, BSs):
    for i in np.arange(BSs):
        index = get_indexes(i, UE_BS_index)
        if len((np.where(scheduling_counter[index] == 0))[0]) != 0:  # hay mas de un minimo
            aux = (np.where(scheduling_counter[index] == 0)[0])  # Indice de todos los minimos

            aux = aux.astype(int)
            index = np.array(index)
            index = index.astype(int)

            idx = (np.where(channel_gain[i, index[aux]] == channel_gain[i, index[aux]].max()))[
                0]  # indice de ganancia maxima
            active_users[i] = index[aux[idx]]  # indice del elegido
            scheduling_counter[index[aux[idx]]] = 1
        else:
            scheduling_counter[index] = scheduling_counter[index] * 0
            aux = (np.where(scheduling_counter[index] == 0)[0])  # Indice de todos los minimos

            aux = aux.astype(int)
            index = np.array(index)
            index = index.astype(int)

            idx = (np.where(channel_gain[i, index[aux]] == channel_gain[i, index[aux]].max()))[
                0]  # indice de ganancia maxima
            active_users[i] = index[aux[idx]]  # indice del elegido
            scheduling_counter[index[aux[idx]]] = 1
    return active_users, scheduling_counter


def get_indexes(x, xs):
    indexes = [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    return indexes


def grid_deployment(Nbs, Rcell):
    cell_position = np.zeros((Nbs, 2))
    if (Nbs > 1):
        theta = np.arange(0, Nbs - 1) * np.pi / 3  # en matlab el vector es vertical
        cell_position[1:, :] = np.sqrt(3) * Rcell * np.concatenate(([np.cos(theta)], [np.sin(theta)]), axis=0).T

    if (Nbs > 7):
        theta = np.arange(start=-np.pi / 6, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x = 3 * Rcell * np.cos(theta)
        y = 3 * Rcell * np.sin(theta)
        theta = np.arange(start=0, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x = np.reshape(np.concatenate(([x], [2 * np.sqrt(3) * Rcell * np.cos(theta)]), axis=0), (2 * theta.size, 1),
                       order='F')
        y = np.reshape(np.concatenate(([y], [2 * np.sqrt(3) * Rcell * np.sin(theta)]), axis=0), (2 * theta.size, 1),
                       order='F')

        if Nbs > 19:
            cell_position[7:19, 0:2] = np.concatenate((x, y), axis=1)
        else:
            cell_position[7:Nbs + 1, 0:2] = np.concatenate((x[0:Nbs - 7], y[0:Nbs - 7]), axis=1)

    if Nbs > 19 and Nbs < 38:
        theta = np.arange(start=-np.arcsin(3 / np.sqrt(21)), stop=(5 / 3) * np.pi, step=np.pi / 3)
        x1 = np.sqrt(21) * Rcell * np.cos(theta)
        y1 = np.sqrt(21) * Rcell * np.sin(theta)
        theta = np.arange(start=-np.arcsin(3 / 2 / np.sqrt(21)), stop=(5 / 3) * np.pi, step=np.pi / 3)
        x2 = np.sqrt(21) * Rcell * np.cos(theta)
        y2 = np.sqrt(21) * Rcell * np.sin(theta)
        theta = np.arange(start=0, stop=(5 / 3) * np.pi, step=np.pi / 3)
        x3 = 3 * np.sqrt(3) * Rcell * np.cos(theta)
        y3 = 3 * np.sqrt(3) * Rcell * np.sin(theta)
        x = np.reshape(np.concatenate(([x1], [x2], [x3]), axis=0), (x1.size + x2.size + x3.size, 1), order='F')
        y = np.reshape(np.concatenate(([y1], [y2], [y3]), axis=0), (y1.size + y2.size + y3.size, 1), order='F')
        cell_position[19:Nbs + 1, 0:2] = np.concatenate((x[0:Nbs - 19], y[0:Nbs - 19]), axis=1)

    x_base_station = cell_position[:, 0]
    y_base_station = cell_position[:, 1]

    return x_base_station, y_base_station, cell_position


def Generate_state(reward_modified, power_modified, sinr_norm_inv, Nbs, C):
    # Need to know if the algorithms its learning, therefore i will use instant rate and power
    indices1 = np.tile(np.expand_dims(np.linspace(0, Nbs - 1, num=Nbs, dtype=np.int32), axis=1), [1, C])
    indices2 = np.argsort(sinr_norm_inv, axis=1)[:, -C:]
    #indices2 = np.argsort(sinr_norm_inv, axis=1)[:, -2:]

    #print(indices1)
    #print(indices2)
    #wait = input('Press enter to continue')

    #rate_state = np.hstack([reward_modified[:, 0:1], reward_modified[indices1, indices2 + 1]])
    #power_state = np.hstack([power_modified[:, 0:1], power_modified[indices1, indices2 + 1]])
    #sinr_norm_inv = sinr_norm_inv[indices1, indices2]
    #s_t = np.hstack([rate_state, power_state, sinr_norm_inv])

    if Nbs >= C:
        rate_state = np.hstack([reward_modified[:, 0:1], reward_modified[indices1, indices2 + 1]])
        power_state = np.hstack([power_modified[:, 0:1], power_modified[indices1, indices2 + 1]])
        sinr_norm_inv = sinr_norm_inv[indices1, indices2]
        s_t = np.hstack([rate_state, power_state, sinr_norm_inv])
    else:
        zeros_aux = np.zeros((Nbs,(C+1)-Nbs))
        rate_state = np.hstack([reward_modified[:, 0:1],zeros_aux, reward_modified[indices1[:,1:Nbs], indices2 + 1]])
        power_state = np.hstack([power_modified[:, 0:1],zeros_aux, power_modified[indices1[:,1:Nbs], indices2 + 1]])
        sinr_norm_inv = np.hstack([zeros_aux, sinr_norm_inv[indices1[:,1:Nbs], indices2]])
        s_t = np.hstack([rate_state, power_state, sinr_norm_inv])

        #print('state \n', s_t)
        #wait = input('Press enter to continue')

    return s_t


def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def user_set(N, Rmin, Rmax):
    '''
    N: Set of points (It should be greater cause points will be removed
    Rmin: Distancia minima del centro al exterior
    Rmax: Radio del hexagono
    '''
    v_x = Rmax * np.cos(np.arange(6) * np.pi / 3)
    v_y = Rmax * np.sin(np.arange(6) * np.pi / 3)

    c_x = Rmax - np.random.random((1, 3 * N)) * 2 * Rmax
    c_y = Rmax - np.random.random((1, 3 * N)) * 2 * Rmax

    IN = inpolygon(c_x, c_y, v_x, v_y)
    # Remove from outside the hexagon
    c_x = c_x[IN]
    c_y = c_y[IN]
    # Remove from inner circle
    aux_dis = np.sqrt(((c_x - 0) ** 2) + ((c_y - 0) ** 2))
    Z = aux_dis > Rmin
    c_x = c_x[Z]
    c_y = c_y[Z]

    idx = np.random.permutation(len(c_x))

    c_x = c_x[idx[1:N]]
    c_y = c_y[idx[1:N]]

    return c_x, c_y


def WMMSE_sum_rate(p_int, H, Pmax, var_noise):
    K = np.size(p_int)
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros(K)
    w = np.zeros(K)
    for i in range(K):
        f[i] = H[i, i] * b[i] / (np.square(H[i, :]) @ np.square(b) + var_noise)
        w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
        if w[i] < 0:
            w[i] = 1
        vnew = vnew + math.log2(w[i])

    VV = np.zeros(100)
    for iter in range(100):
        vold = vnew
        for i in range(K):
            btmp = w[i] * f[i] * H[i, i] / sum(w * np.square(f) * np.square(H[:, i]))
            b[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            f[i] = H[i, i] * b[i] / ((np.square(H[i, :])) @ (np.square(b)) + var_noise)
            w[i] = 1 / (1 - f[i] * b[i] * H[i, i])
            if w[i] < 0:
                w[i] = 1

            vnew = vnew + math.log2(w[i])

        VV[iter] = vnew
        if vnew - vold <= 1e-3:
            break

    p_opt = np.square(b)
    return p_opt


def Simulation_Scenario(BSs, UEs,min_BS_dist, min_UE_BS_dist, width, height):

    # Codigo para extraer valores
    z = [0];
    #while (len(z) < 90):  # añadido porque aveces solo daba ~5 muestras o menos de posicion z
    while (len(z) < 25):  # añadido porque aveces solo daba ~5 muestras o menos de posicion z
        z = poisson_disc_samples(width=width, height=height, r=min_BS_dist)

    # -------------Extraer posicion
    # get every first element in 2 Dimensional List
    x_values = [i[0] for i in z]
    # get every second element in 2 Dimensional List
    y_values = [i[1] for i in z]

    #--------------------------------------------------
    # genera n indices aleatorios sin repetir
    samples = random.sample(range(0, len(z)-1), BSs)

    # extrae los valores de la posicion de los indices extraidos
    x_sample = [x_values[i] for i in samples]
    y_sample = [y_values[i] for i in samples]

    # ----------------UEs Deployment
    x_user = []
    y_user = []
    counter = 0

    while counter < UEs:
        x = random.uniform(0, width)
        y = random.uniform(0, width)
        distances = [math.sqrt(((x-x_sample[i])**2)+((y-y_sample[i])**2)) for i in range(len(x_sample))]
        # Si todas las distancias son mayores a la distancia minima entre UE y BSs
        if all(number > min_UE_BS_dist for number in distances):
            counter += 1
            x_user.append(x)
            y_user.append(y)

    x_sample = np.array(x_sample)   # BSs x-position
    y_sample = np.array(y_sample)   # BSs y-position
    x_user = np.array(x_user)       # UEs x-position
    y_user = np.array(y_user)       # UEs y-position

    return x_sample, y_sample, x_user, y_user


class CellularEnv(Env):
    def __init__(self, Nue, Nbs, Rmin, Rmax, cell_deployment, path_loss, A, intervals,
                 Pmin_dBm, Pmax_dBm, noise_power_dBm, SINR_th, C):

        self.Nue = Nue
        self.Nbs = Nbs
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Pmin_dBm = Pmin_dBm
        self.Pmax_dBm = Pmax_dBm
        self.Pmax = (10 ** ((self.Pmax_dBm - 30) / 10))
        self.Pmin = (10 ** ((self.Pmin_dBm - 30) / 10))
        self.noise_power_dBm = noise_power_dBm
        self.intervals = intervals
        self.path_loss = path_loss
        self.cell_deployment = cell_deployment
        self.scheduling_counter = np.zeros(self.Nue)
        self.active_users = np.zeros(self.Nbs).astype(int)
        self.SINR_th = np.array([SINR_th])
        #self.SINR_cap = np.array(SINR_cap)
        self.A = A
        self.rates = []
        self.power = []
        self.rewards = []
        self.SINR = []
        self.episode_length = 0
        self.reward_rate = np.zeros(self.Nbs, dtype=np.float32)
        self.P_ = np.zeros(self.Nbs)
        self.R_ = np.zeros(self.Nbs)
        # self.power_available = 1e-3 * pow(10., np.linspace(self.Pmin_dBm, self.Pmax_dBm, self.A) / 10.)
        self.power_available = np.hstack([np.zeros((1), dtype=np.float32),
                                          1e-3 * pow(10., np.linspace(self.Pmin_dBm, self.Pmax_dBm, self.A - 1) / 10.)])

        #self.N = (10 ** ((self.noise_power_dBm - 30) / 10))
        self.N = 1e-3*pow(10., self.noise_power_dBm/10)

        self.Tx_power = np.zeros(self.Nbs)
        self.scheme_full_reuse = 1
        self.channel_gain = np.zeros([self.Nbs, self.Nue, self.intervals])
        self.fd = 10
        self.Ts = 20e-3

        # 100K points randomly created, then filtered for a minimum distance on an hexagonal shape
        self.x_user_set, self.y_user_set = user_set(100000, self.Rmin, self.Rmax)

        # Hexagon rotation------------------------------------------------------------------------------------------
        rho = np.sqrt(self.x_user_set ** 2 + self.y_user_set ** 2)
        phi = np.arctan2(self.y_user_set, self.x_user_set)
        self.x_user_set = rho * np.cos(phi - (np.pi / 2))
        self.y_user_set = rho * np.sin(phi - (np.pi / 2))
        #------------------------------------------------------------------------------------------------------------

        print('Numero de posiciones disponibles del conjunto de usuarios: ', len(self.x_user_set))
        # ----------------------------------------------------------------------------------------------------

        # Communication Channel Components ----------------------------------------
        self.K0 = 39
        self.alfa1 = 2
        self.alfa2 = 4
        self.dBP = 100

        #''' Thid is how it needs to be
        self.C = C-1 # where C is the number neighboor BS taked account into the state obsevation
        self.state = np.zeros((self.C, (self.C * 2) + self.C), dtype=np.float32) # {R_,P_,SINR_)
        #'''

        # high = np.ones(self.Nue * 3) * 100
        # low = np.zeros(self.Nue * 3)
        # self.observation_space = Box(low=low, high=high)
        self.action_space = Discrete(self.A)


        # 0. Despliegue de Base Station --------------------------------------------------------------------------------
        if self.cell_deployment == 'poisson':  # Poisson BS deploymen is implemented

            width = 180 #180  #78#102#77#140
            height = 180#180 #90#45#80  # Within Env_DDQN_9_0

            self.x_base_station, self.y_base_station, x_user, y_user = Simulation_Scenario(self.Nbs, self.Nue,
                                                                                 10, 1, width, height)
            self.cell_position = np.vstack((self.x_base_station, self.y_base_station)).T
            # Despliegue en celda circular ----------------------------------------------------------------------------
            user_position = np.zeros((self.Nue, 2))  # User deployment, one UE per BS *
            R = np.random.uniform(low=self.Rmin, high=self.Rmax, size=(1, self.Nue))
            angle = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(1, self.Nue))
            user_position[:, 0] = R * np.cos(angle)
            user_position[:, 1] = R * np.sin(angle)
            user_position = self.cell_position + user_position
            x_user = user_position[:, 0]
            y_user = user_position[:, 1]
        else:                            # Grid BS deployment is implemented
            self.x_base_station, self.y_base_station, self.cell_position = grid_deployment(self.Nbs, self.Rmax)
            #-----------------------------------------------------------------------------------------------------------
            # 0. Despliegue de usuarios Dentro del FALSO hexagono(Girado)
            user_position = np.zeros((self.Nue, 2))  # User deployment, one UE per BS *
            idx = np.random.choice(np.arange(len(self.x_user_set)), self.Nue, replace=False)
            user_position[:, 0] = self.x_user_set[idx]
            user_position[:, 1] = self.y_user_set[idx]
            user_position = self.cell_position + user_position
            x_user = user_position[:, 0]
            y_user = user_position[:, 1]
            #-----------------------------------------------------------------------------------------------------------

        # 1. Calcular la distancia entre estaciones base y usuarios
        distances = BS_UE_distances(self.x_base_station, self.y_base_station, x_user, y_user)
        # 2. Asociar por la distancia mínima y Calcular la carga de cada estacion base
        ''' Por el momento esta parte se va a omitir para que no siga dando errores
        UE_BS_index, BS_load = association(distances, self.Nbs, self.Nue)
        self.UE_BS_index = UE_BS_index
        '''
        self.UE_BS_index = np.arange(self.Nue)
        # Initialization
        # scheduling_counter = np.zeros(len(self.UE_BS_index))
        # 3. Calcular el path loss para el canal de transmision y los canales interferentes

        lognormal = np.random.lognormal(sigma=8, size=(self.Nbs, self.Nue))
        path_loss = lognormal * pow(10., -(path_loss_LTE(distances)) / 10.)
        channel_set = np.zeros([self.Nbs, self.Nue, self.intervals])
        pho = np.float32(scipy.special.k0(2 * np.pi * self.fd * self.Ts))
        channel_set[:, :, 0] = np.sqrt(
            0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2 + np.random.randn(self.Nbs, self.Nue) ** 2))
        for i in range(1, int(intervals)):
            channel_set[:, :, i] = channel_set[:, :, i - 1] * pho + np.sqrt(
                (1. - pho ** 2) * 0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2
                                         + np.random.randn(self.Nbs, self.Nue) ** 2))
        self.channel_gain = np.square(channel_set) * np.tile(np.expand_dims(path_loss, axis=2), [1, 1, int(intervals)])

        # This needs to be modified, we only need the active user index
        if self.scheme_full_reuse == 1:
            self.active_users, self.scheduling_counter = scheduling_per_access_point(
                self.UE_BS_index, self.scheduling_counter, self.channel_gain[:, :, 0], self.active_users, self.Nbs)
        # ------------------------------------------------------------------------------------------------------------

        # ------ Formating for calculations and manipulation -----------------------------------------------------------
        channel_modified, power_modified, reward_modified = np.zeros((self.Nbs, self.Nbs)), \
                                                            np.zeros((self.Nbs, self.Nbs)), np.zeros((self.Nbs, self.Nbs))
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            channel_modified[i_agent_BS] = np.roll(self.channel_gain[self.active_users, idy, 0], -idx)  # initialization
            power_modified[i_agent_BS, :] = np.roll(self.P_, -i_agent_BS)
            reward_modified[i_agent_BS, :] = np.roll(self.R_, -i_agent_BS)
        # Needed for order by priority and to avoid the small values*
        sinr_norm_inv = channel_modified[:, 1:] / np.tile(channel_modified[:, 0:1], [1, Nbs - 1])
        sinr_norm_inv = np.log(1 + sinr_norm_inv) / np.log(2)  # log representation
        # ------ Formating for calculations and manipulation -----------------------------------------------------------

        self.state = Generate_state(reward_modified=reward_modified,
                                    power_modified=power_modified, sinr_norm_inv=sinr_norm_inv, Nbs=self.Nbs, C=self.C)

        # For Validation ---------------------------------------------------------------------------------------------
        '''
        This could be changed to a Validation initialization function to save time and for testing
        '''
        ## Validation set for testing 10,10 and 10 scenarios for Lvl1, Lv2 and Lvl 3 Respectively
        Validation_1 = np.loadtxt('ValidationTest_Lvl_1.dat')
        Validation_2 = np.loadtxt('ValidationTest_Lvl_2.dat')
        Validation_3 = np.loadtxt('ValidationTest_Lvl_3.dat')

        self.Validation_set= np.vstack((Validation_1,Validation_2,Validation_3))

        np.random.seed(10)

        Val_scenarios = 15

        self.val_cell_position = np.zeros((self.Nbs,self.Nue, Val_scenarios))

        self.val_UE_BS_index = np.copy(self.UE_BS_index)
        self.val_channel_gain = np.zeros([self.Nbs, self.Nue, self.intervals, Val_scenarios])
        self.val_state = np.zeros((self.Nbs, self.Nbs*3 - 1, Val_scenarios), dtype=np.float32)  # {R_,P_,SINR_)

        for test_number in np.arange(Val_scenarios):
            if self.cell_deployment == 'poisson':  # Poisson BS deploymen is implemented

                x_base_station, y_base_station, x_user, y_user = Simulation_Scenario(self.Nbs, self.Nue,
                                                                                               10, 1, width, height)
                self.val_cell_position[:,:,test_number] = np.vstack((self.x_base_station, self.y_base_station)).T
                user_position = self.Validation_set[test_number * self.Nbs:(test_number+1) * self.Nbs,:]
                user_position = self.val_cell_position + user_position
                x_user = user_position[:,0]
                y_user = user_position[:,1]
            else:  # Grid BS deployment is implemented
                x_base_station = np.copy(self.x_base_station)
                y_base_station = np.copy(self.y_base_station)
                self.val_cell_position = np.copy(self.cell_position)
                user_position = self.Validation_set[test_number * self.Nbs:(test_number+1) * self.Nbs,:]
                user_position = self.val_cell_position + user_position
                x_user = user_position[:,0]
                y_user = user_position[:,1]

            distances = BS_UE_distances(x_base_station,y_base_station, x_user, y_user)


            lognormal = np.random.lognormal(sigma=8, size=(self.Nbs, self.Nue))
            path_loss = lognormal * pow(10., -(path_loss_LTE(distances)) / 10.)
            channel_set = np.zeros([self.Nbs, self.Nue, self.intervals])
            pho = np.float32(scipy.special.k0(2 * np.pi * self.fd * self.Ts))
            channel_set[:, :, 0] = np.sqrt(
                0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2 + np.random.randn(self.Nbs, self.Nue) ** 2))
            for i in range(1, int(self.intervals)):
                channel_set[:, :, i] = channel_set[:, :, i - 1] * pho + np.sqrt(
                    (1. - pho ** 2) * 0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2
                                             + np.random.randn(self.Nbs, self.Nue) ** 2))
            channel_gain = np.square(channel_set) * np.tile(np.expand_dims(path_loss, axis=2),
                                                                 [1, 1, int(self.intervals)])

            self.val_channel_gain[:,:,:,test_number] = channel_gain

            # ------ Formating for calculations and manipulation -----------------------------------------------------------
            channel_modified, power_modified, reward_modified = np.zeros((self.Nbs, self.Nbs)), \
                                                      np.zeros((self.Nbs, self.Nbs)), np.zeros((self.Nbs, self.Nbs))
            for i_agent_BS in np.arange(self.Nbs):
                idy = self.active_users[i_agent_BS]
                idx = self.UE_BS_index[idy]
                channel_modified[i_agent_BS] = np.roll(self.val_channel_gain[self.active_users, idy, 0, test_number], -idx)
                power_modified[i_agent_BS, :] = np.roll(self.P_, -i_agent_BS)
                reward_modified[i_agent_BS, :] = np.roll(self.R_, -i_agent_BS)
            # Needed for order by priority and to avoid the small values*
            sinr_norm_inv = channel_modified[:, 1:] / np.tile(channel_modified[:, 0:1], [1, self.Nbs - 1])
            sinr_norm_inv = np.log(1 + sinr_norm_inv) / np.log(2)  # log representation
            # ------ Formating for calculations and manipulation ----------------------------------------------------------

            state = Generate_state(reward_modified=reward_modified,
                                        power_modified=power_modified, sinr_norm_inv=sinr_norm_inv, Nbs=self.Nbs,
                                        C=self.C)
            self.val_state[:,:,test_number] = state

        # Desactivate Random seed--------------------------------------------------------------------------------------
        np.random.seed()
        # -------------------------------------------------------------------------------------------------------------

        # For Validation ---------------------------------------------------------------------------------------------

    def step(self, action):
        # --------------------- Reward Calculation -------------------------------------------------------------------
        for i in np.arange(self.Nbs):  # Action is a vector of len(Nbs)
            self.Tx_power[i] = action_map(action[i], self.power_available)


        for i in np.arange(self.Nbs):
            S = self.channel_gain[i, self.active_users[i], self.episode_length] * self.Tx_power[i]

            pi_aux = 0
            weight = np.ones(self.Nbs,dtype=np.float32)

            # Compute Ck
            # Tomando en cuenta que las estaciones base son los renglones (primer indice) y los usuarios las columnas ... -> shape (Nbs, Nue)
            aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
            Interference = np.sum(self.channel_gain[self.UE_BS_index[aux], i, self.episode_length] * self.Tx_power[aux])
            sinr = S / (Interference + self.N)
            if sinr >= self.SINR_th:  # Minimum SINR Restriction
                C = math.log2(1 + sinr)
            else:
                C = 0

            # Compute Ck\i
            for k in np.arange(self.Nbs-1):
                aux2 = [x for j, x in enumerate(self.active_users) if (j != i) and (j != aux[k]) and (x != None)]
                Interference = np.sum(
                    self.channel_gain[self.UE_BS_index[aux2], i, self.episode_length] * self.Tx_power[aux2])
                sinr2 = S / (Interference + self.N)
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C2 = math.log2(1 + sinr2)
                else:
                    C2 = 0
                pi_aux += weight[i]*(C2-C)

            Reward = weight[i]*C-pi_aux

            self.rates.append(C)  # Todas las tasas concatenadas
            self.rewards.append(Reward)

        # Reward Function (Prueba) -------------------------------------------------------------------------------------
        sumrate = np.sum(self.rates[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs])

        # Equal Reward for all Agents regardless their individual contribution
        self.reward_rate = np.ones (self.Nbs) * \
                           np.sum(self.rewards[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) +
                                                                                self.Nbs])

        # State formating -------------------------------------------------------------------------------------------
        self.power = np.hstack((self.power, self.Tx_power))
        self.P_ = self.power[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        self.R_ = self.rates[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]
        #self.R_ = self.rewards[(self.episode_length * self.Nbs):(self.episode_length * self.Nbs) + self.Nbs]

        # ------ Formating for calculations and manipulation -----------------------------------------------------------
        channel_modified, power_modified, reward_modified = np.zeros((self.Nbs, self.Nbs)), \
                                                            np.zeros((self.Nbs, self.Nbs)), np.zeros((self.Nbs, self.Nbs))
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            channel_modified[i_agent_BS] = np.roll(self.channel_gain[self.active_users, idy, self.episode_length], -idx)
            power_modified[i_agent_BS, :] = np.roll(self.P_, -i_agent_BS)
            reward_modified[i_agent_BS, :] = np.roll(self.R_, -i_agent_BS)
        # Needed for order by priority and to avoid the small values*
        sinr_norm_inv = channel_modified[:, 1:] / np.tile(channel_modified[:, 0:1], [1, self.Nbs - 1])
        sinr_norm_inv = np.log(1 + sinr_norm_inv) / np.log(2)  # log representation
        # ------ Formating for calculations and manipulation ----------------------------------------------------------

        self.state = Generate_state(reward_modified=reward_modified,
                                    power_modified=power_modified, sinr_norm_inv=sinr_norm_inv, Nbs=self.Nbs, C=self.C)

        # Set placeholder for info # Requierement of OPENAI Environments
        info = {}

        self.episode_length += 1
        if self.episode_length >= self.intervals:
            done = True
        else:
            done = False

        return self.state, self.reward_rate, done, info, sumrate

    def reset(self):
        # Reset UE deployment ----------------------------------------------------------------------------------------
        self.scheduling_counter = np.zeros(self.Nue)
        # self.active_users = np.zeros(self.Nbs).astype(int)
        # -- Se mantienen el estado anterior al reiniciar el episodio, ya que no se puede tomar una decisión con 0s---
        self.rates = []
        self.rewards = []
        self.SINR = []
        self.power = []
        # self.reward_rate = 0
        self.reward_rate = np.zeros(self.Nbs, dtype=np.float32)

        # Reiniciados porque asi lo tiene el articulo PA_ICC---------------------------------------------------------
        self.P_ = np.zeros(self.Nbs)
        self.R_ = np.zeros(self.Nbs)
        # Reiniciados porque asi lo tiene el articulo PA_ICC---------------------------------------------------------

        # '''
        # 0. Despliegue de Base Station --------------------------------------------------------------------------------
        if self.cell_deployment == 'poisson':  # Poisson BS deploymen is implemented

            width = 180#102#77  # 140
            height = 180#45  # 80  # Within Env_DDQN_9_0

            self.x_base_station, self.y_base_station, x_user, y_user = Simulation_Scenario(self.Nbs, self.Nue,
                                                                                           10, 1, width, height)
            self.cell_position = np.vstack((self.x_base_station, self.y_base_station)).T
            # Despliegue en celda circular ----------------------------------------------------------------------------
            user_position = np.zeros((self.Nue, 2))  # User deployment, one UE per BS *
            R = np.random.uniform(low=self.Rmin, high=self.Rmax, size=(1, self.Nue))
            angle = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(1, self.Nue))
            user_position[:, 0] = R * np.cos(angle)
            user_position[:, 1] = R * np.sin(angle)
            user_position = self.cell_position + user_position
            x_user = user_position[:, 0]
            y_user = user_position[:, 1]
        else:  # Grid BS deployment is implemented
            self.x_base_station, self.y_base_station, self.cell_position = grid_deployment(self.Nbs, self.Rmax)
            # -----------------------------------------------------------------------------------------------------------
            # 0. Despliegue de usuarios Dentro del FALSO hexagono(Girado)
            user_position = np.zeros((self.Nue, 2))  # User deployment, one UE per BS *
            idx = np.random.choice(np.arange(len(self.x_user_set)), self.Nue, replace=False)
            user_position[:, 0] = self.x_user_set[idx]
            user_position[:, 1] = self.y_user_set[idx]
            user_position = self.cell_position + user_position
            x_user = user_position[:, 0]
            y_user = user_position[:, 1]
            # -----------------------------------------------------------------------------------------------------------

        # 1. Calcular la distancia entre estaciones base y usuarios
        distances = BS_UE_distances(self.x_base_station, self.y_base_station, x_user, y_user)
        # 2. Asociar por la distancia mínima y Calcular la carga de cada estacion base
        self.UE_BS_index = np.arange(self.Nue)

        # Initialization
        # scheduling_counter = np.zeros(len(self.UE_BS_index))
        # 3. Calcular el path loss para el canal de transmision y los canales interferentes

        lognormal = np.random.lognormal(sigma=8, size=(self.Nbs, self.Nue))
        path_loss = lognormal * pow(10., -(path_loss_LTE(distances)) / 10.)
        channel_set = np.zeros([self.Nbs, self.Nue, self.intervals])
        pho = np.float32(scipy.special.k0(2 * np.pi * self.fd * self.Ts))
        channel_set[:, :, 0] = np.sqrt(
            0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2 + np.random.randn(self.Nbs, self.Nue) ** 2))
        for i in range(1, int(self.intervals)):
            channel_set[:, :, i] = channel_set[:, :, i - 1] * pho + np.sqrt(
                (1. - pho ** 2) * 0.5 * (np.random.randn(self.Nbs, self.Nue) ** 2
                                         + np.random.randn(self.Nbs, self.Nue) ** 2))
        self.channel_gain = np.square(channel_set) * np.tile(np.expand_dims(path_loss, axis=2),
                                                             [1, 1, int(self.intervals)])

        self.episode_length = 0

        # ------ Formating for calculations and manipulation -----------------------------------------------------------
        channel_modified, power_modified, reward_modified = np.zeros((self.Nbs, self.Nbs)), \
                                                            np.zeros((self.Nbs, self.Nbs)), np.zeros(
            (self.Nbs, self.Nbs))
        for i_agent_BS in np.arange(self.Nbs):
            idy = self.active_users[i_agent_BS]
            idx = self.UE_BS_index[idy]
            channel_modified[i_agent_BS] = np.roll(self.channel_gain[self.active_users, idy, 0], -idx)
            power_modified[i_agent_BS, :] = np.roll(self.P_, -i_agent_BS)
            reward_modified[i_agent_BS, :] = np.roll(self.R_, -i_agent_BS)
        # Needed for order by priority and to avoid the small values*
        sinr_norm_inv = channel_modified[:, 1:] / np.tile(channel_modified[:, 0:1], [1, self.Nbs - 1])
        sinr_norm_inv = np.log(1 + sinr_norm_inv) / np.log(2)  # log representation
        # ------ Formating for calculations and manipulation ----------------------------------------------------------

        self.state = Generate_state(reward_modified=reward_modified,
                                    power_modified=power_modified, sinr_norm_inv=sinr_norm_inv, Nbs=self.Nbs, C=self.C)

        return self.state


    def validation1_reset(self, test_number): # There are 50 UE deployment positions fixed
        # Reset UE deployment ----------------------------------------------------------------------------------------
        self.scheduling_counter = np.zeros(self.Nue)
        # self.active_users = np.zeros(self.Nbs).astype(int)
        # -- Se mantienen el estado anterior al reiniciar el episodio, ya que no se puede tomar una decisión con 0s---
        self.rates = []
        self.rewards = []
        self.SINR = []
        self.power = []
        # self.reward_rate = 0
        self.reward_rate = np.zeros(self.Nbs, dtype=np.float32)

        # Reiniciados porque asi lo tiene el articulo PA_ICC---------------------------------------------------------
        self.P_ = np.zeros(self.Nbs)
        self.R_ = np.zeros(self.Nbs)
        # Reiniciados porque asi lo tiene el articulo PA_ICC---------------------------------------------------------

        self.episode_length = 0

        self.channel_gain = np.copy(self.val_channel_gain[:,:,:,test_number])
        self.state = np.copy(self.val_state[:,:,test_number])

        return self.state

    def individual_rates(self):
        return self.rates

    def maximum_power(self):
        '''
        Asignacion de potencia maxima
        '''
        Rates_episode1 = []
        Rates1 = []
        Tx_power1 = self.Pmax * np.ones((self.Nbs))  # Potencia Máxima
        for index in range(self.intervals):  # para cada intervalo
            H2 = self.channel_gain[:, :, index]
            # Rates1 = []
            for i in np.arange(self.Nbs):
                S = H2[i, self.active_users[i]] * Tx_power1[i]
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(H2[self.UE_BS_index[aux], i] * Tx_power1[aux])

                # SINR cap -------------------------------------------------------------------------------
                #sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction
                sinr = S / (Interference + self.N)

                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                Rates1.append(C)
            sumrate1 = np.sum(Rates1[(index * self.Nbs):(index * self.Nbs) + self.Nbs])

            Rates_episode1.append(sumrate1)

        #return sum(Rates_episode1) / (self.intervals * self.Nbs)
        return sum(Rates_episode1) / (self.intervals), Rates1

    def random_power(self):
        '''
        Asignacion de potencia aleartoria
        '''
        Rates_episode2 = []
        Rates2 = []
        #Tx_power2 = np.random.uniform(low=self.Pmin, high=self.Pmax, size=self.Nbs)
        Tx_power2 = np.random.uniform(low=0, high=self.Pmax, size=self.Nbs)
        for index in range(self.intervals):  # para cada intervalo
            H2 = self.channel_gain[:, :, index]
            # Rates1 = []
            for i in np.arange(self.Nbs):
                S = H2[i, self.active_users[i]] * Tx_power2[i]
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(H2[self.UE_BS_index[aux], i] * Tx_power2[aux])

                # SINR cap -------------------------------------------------------------------------------
                #sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction
                sinr = S / (Interference + self.N)

                # C = math.log2(1+sinr)
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                Rates2.append(C)
            sumrate2 = np.sum(Rates2[(index * self.Nbs):(index * self.Nbs) + self.Nbs])
            Rates_episode2.append(sumrate2)
        #return sum(Rates_episode2) / (self.intervals * self.Nbs)
        return sum(Rates_episode2) / (self.intervals), Rates2

    '''
    def WMMSE_power(self):
        #
        #Source: https://github.com/mengxiaomao/PA_ICC/blob/121832fffc4031ac548f1f5c7477b45cf1e9e148/PA_alg.py#L44
        #
        Rates_episode1 = []
        Rates1 = []
        p_array = np.zeros((self.Nbs, self.Nbs), dtype=np.int32)
        Z = np.arange(self.Nbs, dtype=np.int32)
        for index in range(self.intervals):  # para cada intervalo
            # Step 1: Change the channel format
            channel_modified = np.zeros((self.Nbs, self.Nbs))
            for i_agent_BS in np.arange(self.Nbs):
                idy = self.active_users[i_agent_BS]
                idx = self.UE_BS_index[idy]
                channel_modified[i_agent_BS] = np.roll(self.channel_gain[self.active_users, idy, index],
                                                       -idx)  # initialization
                p_array[i_agent_BS] = np.roll(Z, -idx)

            H2 = channel_modified

            hkk = np.sqrt(H2[:, 0])
            v = np.random.rand(self.Nbs)  # maxP*np.ones((N))

            # V_extend = np.hstack([v, np.zeros(((self.M - self.N + 1)), dtype=dtype)])
            V_extend = np.hstack([v, np.zeros(((self.Nue - self.Nbs + 1)), dtype=np.float32)])
            V = np.reshape(V_extend[p_array], [self.Nbs, self.Nue])
            u = hkk * v / (np.sum(H2 * V ** 2, axis=1) + self.N)
            w = 1. / (1. - u * hkk * v)
            C = np.sum(w)
            W_extend = np.hstack([w, np.zeros((self.Nue - self.Nbs + 1), dtype=np.float32)])
            W = np.reshape(W_extend[p_array], [self.Nbs, self.Nue])
            U_extend = np.hstack([u, np.zeros((self.Nue - self.Nbs + 1), dtype=np.float32)])
            U = np.reshape(U_extend[p_array], [self.Nbs, self.Nue])
            for cou in range(100):
                C_last = C
                v = w * u * hkk / np.sum(W * U ** 2 * H2, axis=1)
                v = np.minimum(np.sqrt(self.Pmax), np.maximum(1e-10 * np.random.rand(self.Nbs), v))
                V_extend = np.hstack([v, np.zeros((self.Nue - self.Nbs + 1), dtype=np.float32)])
                V = np.reshape(V_extend[p_array], [self.Nbs, self.Nue])
                u = hkk * v / (np.sum(H2 * V ** 2, axis=1) + self.N)
                w = 1. / (1. - u * hkk * v)
                C = np.sum(w)
                if np.abs(C_last - C) < 1e-3:
                    break
                W_extend = np.hstack([w, np.zeros((self.Nue - self.Nbs + 1), dtype=np.float32)])
                W = np.reshape(W_extend[p_array], [self.Nbs, self.Nue])
                U_extend = np.hstack([u, np.zeros((self.Nue - self.Nbs + 1), dtype=np.float32)])
                U = np.reshape(U_extend[p_array], [self.Nbs, self.Nue])
            P = v ** 2
            # -------------------------------------------------------------------------------------------------------
            Tx_power1 = P
            # Rates1 = []
            for i in np.arange(self.Nbs):
                S = H2[i, self.active_users[i]] * Tx_power1[i]
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(H2[self.UE_BS_index[aux], i] * Tx_power1[aux])
                sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction
                # C = math.log2(1 + sinr)
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                Rates1.append(C)
            sumrate1 = np.sum(Rates1[(index * self.Nbs):(index * self.Nbs) + self.Nbs])
            Rates_episode1.append(sumrate1)
        return sum(Rates_episode1) / (self.intervals * self.Nbs)
    '''
    def WMMSE_power2(self):
        '''
        Asignacion de potencia WMMSE
        '''
        Rates_episode3 = []
        Rates3 = []
        for index in range(self.intervals):  # para cada intervalo
            H2 = self.channel_gain[:, :, index]
            Tx_power3 = WMMSE_sum_rate(self.Pmax * np.ones(self.Nbs), np.sqrt(H2), self.Pmax, self.N)

            #Tx_power3[Tx_power3 <= self.Pmin] = 0

            for i in np.arange(self.Nbs):
                S = H2[i, self.active_users[i]] * Tx_power3[i]
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(H2[self.UE_BS_index[aux], i] * Tx_power3[aux])

                # SINR cap -------------------------------------------------------------------------------
                #sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction
                sinr = S / (Interference + self.N)

                # C = math.log2(1+sinr)
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                Rates3.append(C)
            sumrate3 = np.sum(Rates3[(index * self.Nbs):(index * self.Nbs) + self.Nbs])
            Rates_episode3.append(sumrate3)
        #return sum(Rates_episode3) / (self.intervals)
        return sum(Rates_episode3) / self.intervals, Rates3


    def FP_power(self):
        # For initialization
        H2=np.zeros((self.Nbs,self.Nue))
        P_matrix= np.zeros((self.Nbs, self.Nue))
        W= np.ones(self.Nbs)
        Rates_episode4 = []
        Rates4 = []
        for index in range(self.intervals):  # para cada intervalo
            P=np.random.uniform(0, self.Pmax, self.Nbs)
            #P=np.ones(self.Nbs) * (self.Pmax/2)
            #P = np.ones(self.Nbs) *self.Pmax
            for i_agent_BS in np.arange(self.Nbs):
                idy = self.active_users[i_agent_BS]
                idx = self.UE_BS_index[idy]
                # Formating for operations -------------------------------------------------------------------------
                H2[i_agent_BS, :] = np.roll(self.channel_gain[self.active_users, idy, index], -idx)  # initialization
                P_matrix[i_agent_BS, :] = np.roll(P, -i_agent_BS)
                # Formating for operations -------------------------------------------------------------------------
            g_ii = H2[:,0]
            for cou in range(100):
                P_last = np.copy(P_matrix[:,0])
                gamma = g_ii * P_matrix[:, 0] / (np.sum(H2[:, 1:] * P_matrix[:, 1:], axis=1) + self.N)
                y = np.sqrt(W*(1.+gamma) * g_ii * P_matrix[:,0]) / (np.sum(H2 * P_matrix, axis=1) + self.N)
                y_j = np.tile(np.expand_dims(y, axis=1), [1,self.Nbs]) #*******************************************

                #for i_agent_BS in np.arange(self.Nbs):
                #    y_j[i_agent_BS, :] = np.roll(y, -i_agent_BS)

                P = np.minimum(self.Pmax, np.square(y) * W *(1.+gamma) * g_ii / np.sum(np.square(y_j)*H2, axis=1))
                if np.linalg.norm(P_last - P) < 1e-3:
                    break
                for i_agent_BS in np.arange(self.Nbs):
                    idy = self.active_users[i_agent_BS]
                    P_matrix[i_agent_BS, :] = np.roll(P, -i_agent_BS)

            # End of FP algorithm
            Tx_power4 = P
            H2 = self.channel_gain[:, :, index]

            for i in np.arange(self.Nbs):
                S = H2[i, self.active_users[i]] * Tx_power4[i]
                aux = [x for j, x in enumerate(self.active_users) if (j != i) and (x != None)]
                Interference = np.sum(H2[self.UE_BS_index[aux], i] * Tx_power4[aux])

                # SINR cap -------------------------------------------------------------------------------
                #sinr = np.minimum(S / (Interference + self.N), self.SINR_cap)  # Maximum SINR Restriction
                sinr = S / (Interference + self.N)

                # C = math.log2(1+sinr)
                if sinr >= self.SINR_th:  # Minimum SINR Restriction
                    C = math.log2(1 + sinr)
                else:
                    C = 0
                Rates4.append(C)
            sumrate4 = np.sum(Rates4[(index * self.Nbs):(index * self.Nbs) + self.Nbs])
            Rates_episode4.append(sumrate4)
        #return sum(Rates_episode4) / (self.intervals * self.Nbs)
        return sum(Rates_episode4) / (self.intervals), Rates4
