import logging
import math
import random
from server import Server
from sklearn.cluster import KMeans
from threading import Thread
import utils.dists as dists  # pylint: disable=no-name-in-module

from sklearn.decomposition import PCA
import time
from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np


class DQNTrainServer(Server):
    """Federated learning server that uses Double DQN for device selection."""

    def __init__(self, config, case_name):
        
        super().__init__(config,case_name)

        self.memory = deque(maxlen=self.config.dqn.memory_size)
        self.nA = self.config.clients.total
        self.episode = self.config.dqn.episode
        self.max_steps = self.config.dqn.max_steps
        self.target_update = self.config.dqn.target_update
        self.batch_size = self.config.dqn.batch_size
        self.gamma = self.config.dqn.gamma
        self.pca_n_components = 100 # number of components to use for PCA
        self.pca = None

        self.dqn_model = self._build_model()
        self.target_model = self._build_model()

        print("nA =", self.nA)
        # self.total_steps = 0

    def _build_model(self):
        layers = self.config.dqn.hidden_layers # hidden layers

        # (all clients weight + server weight) * pca_n_components, flattened to 1D
        input_size = (self.config.clients.total + 1) * self.pca_n_components 

        states = Input(shape=(input_size,))
        z = states
        for l in layers:
            z = Dense(l, activation='linear')(z)

        q = Dense(self.config.clients.total, activation='softmax')(z)

        model = Model(inputs=[states], outputs=[q])
        model.compile(optimizer=Adam(lr=self.config.dqn.learning_rate), loss=huber_loss)

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.dqn_model.get_weights())

    def epsilon_greedy(self, state, epsilon_current):

        nA = self.nA
        epsilon = epsilon_current #  the probability of choosing a random action
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(self.dqn_model.predict([[state]])[0])
        action_probs[best_action] += (1 - epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def create_greedy_policy(self):

        def policy_fn(state):
            return np.argmax(self.dqn_model.predict([[state]])[0])

        return policy_fn


    def train_episode(self, epsilon_current):

        # Reset the environment
        
        # state = self.get_model_weights_for_state(self.clients)
        # state = np.reshape(state, (1,10000))

        state = self.dqn_reset_state() #++ reset the state at beginning of each episode, randomly select k devices to reset the states

        for t in range(self.max_steps):
            action = self.epsilon_greedy(state, epsilon_current)
            next_state, reward, done = self.step(action) #++ during training, pick a client for next communication round
            self.memorize(state, action, reward, next_state, done)
            self.replay() # sample a mini-batch from the replay buffer to train the DQN model
            state = next_state

            if done:
                break

            if t % self.target_update == 0:
                self.update_target_model()        


    def replay(self):
        
        if len(self.memory) > self.batch_size:
            sample_batch = random.sample(self.memory, self.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in sample_batch:
                states.append(state)
                # need to use the model to predict the q values
                q = self.dqn_model.predict([[state]])[0]

                # then update the experiencd action value using the target model while keeping the other action values the same
                if done:
                    q[action] = reward
                else:
                    q[action] = reward + self.gamma * np.max(self.target_model.predict([[next_state]])[0])

                target_q.append(q)

            states = np.array(states)
            target_q = np.array(target_q)

            self.dqn_model.fit(states, target_q, epochs=1, verbose=0)


    # Run multiple episodes of training for DQN
    def run(self):

        # initial profiling on all clients to get initial pca weights for each client and server model
        self.profile_all_clients()

        # write out the Episode, reward, round, accuracy 
        with open(self.case_name + '_rewards.csv', 'w') as f:
            f.write('Episode,Reward, Round, Accuracy\n')

        for i_episode in range(self.episode):

            # calculate the epsilon value for the current episode
            epsilon_current = self.config.epsilon_initial * pow(self.config.dqn.epsilon_decay, i_episode)
            epsilon_current = max(self.config.dqn.epsilon_min, epsilon_current)

            reward, com_round, final_acc = self.train_episode(epsilon_current) # return the reward and round number for the episode

            print("Episode: {}/{}, reward: {}, com_round: {}, final_acc: {:.4f}".format(i_episode, self.episode, reward, com_round, final_acc))
            with open(self.case_name + '_rewards.csv', 'a') as f:
                f.write('{},{},{},{}\n'.format(i_episode, reward, round, final_acc))
        
        print("\nTraining finished!")

        # Continue federated learning
        # super().run()

    # Federated learning phases
    def dqnselection(self,action):
        # # Select devices to participate in round
        # clients_per_round = self.config.clients.per_round
        # cluster_labels = self.clients.keys()
        #
        # # Generate uniform distribution for selecting clients
        # dist = dists.uniform(clients_per_round, len(cluster_labels))
        #
        # # Select clients from KMeans clusters
        # sample_clients = []
        # for i, cluster in enumerate(cluster_labels):
        #     # Select clients according to distribution
        #     if len(self.clients[cluster]) >= dist[i]:
        #         k = dist[i]
        #     else:  # If not enough clients in cluster, use all avaliable
        #         k = len(self.clients[cluster])
        #
        #     sample_clients.extend(random.sample( # random sample
        #         self.clients[cluster], k))
        #
        #  # Shuffle selected sample clients
        # random.shuffle(sample_clients)
        # sample_clients = []
        # sample_client = self.epsilon_greedy(state)
        # sample_clients.append(sample_client)
        sample_clients_list = [self.clients[action]]

        return sample_clients_list

    def calculate_reward(self,accuracy_this_round):
        target_accuracy = self.config.federated_learning.target_accuracy
        xi = self.config.dqn.reward_xi # in article set to 64
        reward = xi**(accuracy_this_round - target_accuracy) -1
        return reward

    def step(self, action):

        accuracy_this_round = self.round(action)
        reward =self.calculate_reward(accuracy_this_round)

        next_state = self.get_model_weights_for_state(self.clients)

        # determine if the episode is done based on if reaching the target testing accuracy        
        if accuracy_this_round >= self.config.dqn.target_accuracy:
            done = True
        else:
            done = False

        return reward, next_state, done


    def get_model_weights_for_state(self, clients, boot=False):

        # Configure clients to train on local data
        self.configuration(clients)

        # Train on local data for profiling purposes
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports = self.reporting(clients)

        # Extract weights from reports
        # reduced_weights = [self.getPCAWeight(report.weights) for report in reports]
        clients_weights = [self.flatten_weights(report.weights) for report in reports]
        # print("clients_weights: ", clients_weights)
        print("type of clients_weights[0]: ", type(clients_weights[0]))
        print("shape of clients_weights[0]: ", clients_weights[0].shape)

        if boot: # first time to initialize the PCA model
            # build the PCA transformer
            t_start = time.time()
            print("Start building the PCA transformer...")
            self.pca = PCA(n_components=self.pca_n_components)
            clients_weights_pca = self.pca.fit_transform(clients_weights)
            t_end = time.time()
            print("Built PCA transformer, time: {:.2f} s", t_end - t_start)
        
        else: # directly use the pca model to transform the weights
            clients_weights_pca = self.pca.transform(client_weights)

        # print("clients_weights_pca: ", clients_weights_pca)
        print("type of clients_weights_pca[0]: ", type(clients_weights_pca[0]))
        print("shape of clients_weights_pca[0]: ", clients_weights_pca[0].shape)

        # get server model updated weights based on reports from clients
        server_weights = self.aggregation(reports)
        server_weights_pca = self.pca.transform(server_weights)

        print("server_weights: ", server_weights)
        print("server_weights_pca: ", server_weights_pca)
        print("shape of server_weights_pca: ", server_weights_pca.shape)

        stop

        # weight_vecs = []
        # for weight in reduced_weights:
        #     weight_vecs.extend(weight.flatten().tolist())

        return np.array(clients_weights_pca), server_weights_pca
        # return self.flatten_weights(weights)

    """
    def getPCAWeight(self,weight):
        weight_flatten_array = self.flatten_weights(weight)
       ## demision = int(math.sqrt(weight_flatten_array.size))
        # weight_flatten_array = np.abs(weight_flatten_array)
        # sorted_array = np.sort(weight_flatten_array)
        # reverse_array = sorted_array[::-1]

        demision = weight_flatten_array.size
        weight_flatten_matrix = np.reshape(weight_flatten_array,(10,int(demision/10)))
        
        pca = PCA(n_components=10)
        pca.fit_transform(weight_flatten_matrix)
        newWeight = pca.transform(weight_flatten_matrix)
        # newWeight = reverse_array[0:100]

        return  newWeight
    """

    """
    def prefs_to_weights(self):
        prefs = [client.pref for client in self.clients]
        return list(zip(prefs, self.get_model_weights_for_state(self.clients)))
    """
    """
    def profiling(self, clients):
        # return a list of pca weights for each client
        clients_weights, server_weights = self.get_model_weights_for_state(clients)

        return clients_weights, server_weights
    """

    # Server operations
    def profile_all_clients(self):

        # all clients send updated weights to server, the server will do FedAvg
        # And then run  PCA and store the transformed weights

        print("Start profiling all clients...")

        assert len(self.clients)== self.config.clients.total

        # Perform profiling on all clients
        clients_weights_pca, server_weights_pca = self.get_model_weights_for_state(self.clients, True)

        # save the initial pca weights for each client + server 
        self.pca_weights_clientserver_init = clients_weights_pca + server_weights_pca #++

        # save a copy for later update in DQN training episodes
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy() 


    def add_client(self):
        # Add a new client to the server
        raise NotImplementedError
