import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

''' Differences from DDQN_9_0
Epsilon decrement automatized
'''


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)


        #self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.uint8)


        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.n_actions = n_actions

    def store_transition(self, state, action, reward, state_, done, agents):
        for i_agent in range(agents):
            index = self.mem_cntr % self.mem_size
            self.state_memory[index] = state[i_agent]
            self.new_state_memory[index] = state_[i_agent]
            actions = np.zeros(self.n_actions)
            actions[action[i_agent]] = 1.0
            self.action_memory[index] = actions
            self.reward_memory[index] = reward[i_agent]
            self.terminal_memory[index] = done
            self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)
        # batch = np.random.choice(max_mem, batch_size, replace=True)

        #print('Contador de memoria', self.mem_cntr)
        #print('Tamaño memoria', self.mem_size)
        #print(batch.max())
        #print(self.state_memory[batch])
        #wait = input('Press enter to continue')

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        #print('ACCIONES DEL BUFFER', actions)
        #wait=input('Press enter to continue')
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Pruebas: {(64, 64), (128, 64), (256, 64), (512, 64)}
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        #self.V = nn.Linear(256, 1)
        self.A = nn.Linear(64, n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat1 = F.relu(self.fc2(flat1))
        #V = self.V(flat1)
        A = self.A(flat1) # A is the action set
        return A
        #return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    # def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
    #             mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
    #             replace=1000, chkpt_dir='tmp/dueling_ddqn'):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min, eps_dec,
                 replace, chkpt_dir, q_eval_name, q_next_name):
        # path='D:\PyCharmCommunityEdition2020.2.1\Projects\GoogleColab'
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.initial_eps = np.copy(epsilon)
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        # self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.memory = ReplayBuffer(mem_size, input_dims, self.n_actions)



        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=q_eval_name,
                                          chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=q_next_name,
                                          chkpt_dir=self.chkpt_dir)
        '''
        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_eval',
                                          chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name='lunar_lander_dueling_ddqn_q_next',
                                          chkpt_dir=self.chkpt_dir)
        '''

    # '''
    def choose_action(self, observation, random_epsilon):
        # if np.random.random() > self.epsilon:
        if random_epsilon > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            #_, advantage = self.q_eval.forward(state)
            advantage = self.q_eval.forward(state)

            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    # '''

    def choose_action_test(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
       # _, advantage = self.q_eval.forward(state)
        advantage = self.q_eval.forward(state)
        action = T.argmax(advantage).item()

        return action

    def store_transition(self, state, action, reward, state_, done, agents):
        self.memory.store_transition(state, action, reward, state_, done, agents)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    #def decrement_epsilon(self):
    #    self.epsilon = self.epsilon - self.eps_dec \
    #        if self.epsilon > self.eps_min else self.eps_min

    def decrement_epsilon(self, warm_episodes, i_episode, episodes):
        self.epsilon = self.initial_eps - (i_episode - warm_episodes) * (self.initial_eps-self.eps_min) / (episodes - warm_episodes) \
        if self.epsilon > self.eps_min else self.eps_min

    def decrement_learning(self,learning_rate, i_episode, episodes, warm_episodes):
        self.lr = learning_rate * (0.1 ** ((i_episode - warm_episodes) / (episodes - warm_episodes)))

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: # ORIGINAL
            return
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network() # Update Network

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        action_values = np.array(self.action_space, dtype=np.int32)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.forward(states).to(self.q_eval.device)
        q_target = q_eval.clone()
        q_next = self.q_eval.forward(states_).to(self.q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = rewards + \
                                                self.gamma * T.max(q_next, dim=1)[0] * dones

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

    def reset_epsilon(self, epsilon):
        self.epsilon = epsilon
