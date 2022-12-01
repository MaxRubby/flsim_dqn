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
        # number of components to use for PCA, notice here pca_n_components should be smaller than the total number of clients!!!
        self.pca_n_components = min(100, self.config.clients.total)  
        self.pca = None

        self.dqn_model = self._build_model()
        self.target_model = self._build_model()

        self.pca_weights_clientserver_init = None
        self.pca_weights_clientserver = None

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

        q = Dense(self.config.clients.total, activation='linear')(z) # here use linear activation function to predict the q values for each action/client

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
        best_action = np.argmax(self.dqn_model.predict([state])[0])
        action_probs[best_action] += (1 - epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def create_greedy_policy(self):

        def policy_fn(state):
            return np.argmax(self.dqn_model.predict([state])[0])

        return policy_fn
    

    def dqn_round(self, random=False, action=0):
        # default: select the 

        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        if random:
            sample_clients = self.selection()
            print("randomly select clients:", sample_clients)
        else:
            sample_clients = self.dqn_selection(action)
            print("dqn select clients:", sample_clients)

        sample_clients_ids = [client.client_id for client in sample_clients]   

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Receive client updates
        reports = self.reporting(sample_clients) # list of weight tensors

        # client weights pca
        clients_weights = [self.flatten_weights(report.weights) for report in reports] # list of numpy arrays
        clients_weights = np.array(clients_weights) # convert to numpy array
        clients_weights_pca = self.pca.transform(clients_weights)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)
        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # server weight pca
        server_weights = [self.flatten_weights(updated_weights)]
        server_weights = np.array(server_weights)
        server_weights_pca = self.pca.transform(server_weights)

        # update the weights of the selected devices and server to corresponding client id 
        # return next_state
        for i in range(len(sample_clients_ids)):
            self.pca_weights_clientserver[sample_clients_ids[i]] = clients_weights_pca[i]
        
        self.pca_weights_clientserver[-1] = server_weights_pca[0]

        next_state = self.pca_weights_clientserver.flatten()
        print("next_state.shape:", next_state.shape)
        next_state = next_state.tolist()
        
    
        # Test global model accuracy
        if self.config.clients.do_test:  # Get average test accuracy from client reports
            print('Get average accuracy from client reports')
            accuracy = self.accuracy_averaging(reports)

        else:  # Test updated model on server using the aggregated weights
            print('Test updated model on server')
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))

        # testing accuracy, updated pca_weights_clientserver
        return accuracy, next_state


    def dqn_reset_state(self):

        # randomly select k devices to conduct 1 round of FL to reset the states
        # only update the weights of the selected devices in self.pca_weights_clientserver_init

        # copy over again
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

        # randomly select k devices, update the weights of the selected devices and server to get next_state
        accuracy, new_state = self.dqn_round(random=True) # updated self.pca_weights_clientserver

        return new_state

    def choose_action(self, state):
        
        # predict the q values for each action given the current state
        print("DQN choose action")
        q_values = self.dqn_model.predict([state], verbose=0)[0]

        print("q_values:", q_values)
        # use a softmax function to convert the q values to probabilities
        probs = np.exp(q_values) / np.sum(np.exp(q_values))
        print("probs:", probs)

        # add small value to each probability to avoid 0 probability
        #probs = probs + 0.000001

        # choose an action based on the probabilities
        action = np.random.choice(self.nA, p=probs)

        return action


    def train_episode(self, episode_ct, epsilon_current):
        
        # state = self.get_model_weights_for_state(self.clients)
        # state = np.reshape(state, (1,10000))

        # reset the state at beginning of each episode, randomly select k devices to reset the states
        state = self.dqn_reset_state() #++ reset the state at beginning of each episode, randomly select k devices to reset the states

        total_reward = 0
        com_rounds = 0
        final_acc = 0
        for t in range(self.max_steps):
            
            # action = self.epsilon_greedy(state, epsilon_current)
            action = self.choose_action(state)
            next_state, reward, done, acc = self.step(action) #++ during training, pick a client for next communication round
            print("episode_ct:", episode_ct, "step:", t, "acc:", acc, "action:", action, "reward:", reward, "done:", done)
            total_reward += reward
            com_rounds += 1
            final_acc = acc

            self.memorize(state, action, reward, next_state, done)
            self.replay() # sample a mini-batch from the replay buffer to train the DQN model
            state = next_state

            if done:
                break

            if t % self.target_update == 0:
                self.update_target_model()

        return total_reward, com_round, final_acc        


    def replay(self):
        
        if len(self.memory) > self.batch_size:
            print("Replaying...")
            sample_batch = random.sample(self.memory, self.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in sample_batch:
                states.append(state)
                # need to use the model to predict the q values
                q = self.dqn_model.predict([state], verbose=0)[0]
                # print("rest")

                # then update the experiencd action value using the target model while keeping the other action values the same
                if done:
                    q[action] = reward
                else:
                    q[action] = reward + self.gamma * np.max(self.target_model.predict([next_state], verbose=0)[0])

                target_q.append(q)

            states = np.array(states)
            target_q = np.array(target_q)

            print("Fit dqn_model")
            self.dqn_model.fit(states, target_q, epochs=1, verbose=0)
            print("Replay done.")


    # Run multiple episodes of training for DQN
    def run(self):

        # initial profiling on all clients to get initial pca weights for each client and server model
        self.profile_all_clients()

        # write out the Episode, reward, round, accuracy 
        fn = self.config.dqn.rewards_log
        print("Reards logs written to:", fn)
        with open(fn, 'w') as f:
            f.write('Episode,Reward,Round,Accuracy\n')

        for i_episode in range(self.episode):

            t_start = time.time()
            # calculate the epsilon value for the current episode
            epsilon_current = self.config.dqn.epsilon_initial * pow(self.config.dqn.epsilon_decay, i_episode)
            epsilon_current = max(self.config.dqn.epsilon_min, epsilon_current)

            total_reward, com_round, final_acc = self.train_episode(i_episode+1, epsilon_current)

            t_end = time.time()
            print("Episode: {}/{}, reward: {}, com_round: {}, final_acc: {:.4f}, time: {:.2f} s".format(i_episode+1, self.episode, total_reward, com_round, final_acc, t_end - t_start))
            with open(fn, 'a') as f:
                f.write('{},{},{},{}\n'.format(i_episode, total_reward, com_round, final_acc))
        
        print("\nTraining finished!")

        # save trained model to h5 file
        self.dqn_model.save(self.config.dqn.saved_model)
        print("DQN model saved to:", self.config.dqn.saved_model)


    # Federated learning phases
    def dqn_selection(self, action):

        sample_clients_list = [self.clients[action]]

        return sample_clients_list

    def calculate_reward(self, accuracy_this_round):
        
        target_accuracy = self.config.fl.target_accuracy
        xi = self.config.dqn.reward_xi # in article set to 64
        reward = xi**(accuracy_this_round - target_accuracy) -1

        return reward

    def step(self, action):

        accuracy, next_state = self.dqn_round(random=False, action=action) 
        
        # calculate the reward based on the accuracy and the number of communication rounds
        reward =self.calculate_reward(accuracy)

        # determine if the episode is done based on if reaching the target testing accuracy        
        if accuracy >= self.config.fl.target_accuracy:
            done = True
        else:
            done = False

        return next_state, reward, done, accuracy


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
        clients_weights = [self.flatten_weights(report.weights) for report in reports] # list of numpy arrays
        clients_weights = np.array(clients_weights) # convert to numpy array

        # print("clients_weights: ", clients_weights)
        # print("type of clients_weights[0]: ", type(clients_weights[0]))
        print("shape of clients_weights: ", clients_weights.shape)

        if boot: # first time to initialize the PCA model
            # build the PCA transformer
            t_start = time.time()
            print("Start building the PCA transformer...")
            self.pca = PCA(n_components=self.pca_n_components)
            clients_weights_pca = self.pca.fit_transform(clients_weights)
            t_end = time.time()
            print("Built PCA transformer, time: {:.2f} s".format(t_end - t_start))
        
        else: # directly use the pca model to transform the weights
            clients_weights_pca = self.pca.transform(clients_weights)

        # print("clients_weights_pca: ", clients_weights_pca)
        # print("type of clients_weights_pca[0]: ", type(clients_weights_pca[0]))
        print("shape of clients_weights_pca: ", clients_weights_pca.shape)

        # get server model updated weights based on reports from clients
        server_weights = [self.flatten_weights(self.aggregation(reports))]
        server_weights = np.array(server_weights)
        server_weights_pca = self.pca.transform(server_weights)

        # print("server_weights: ", server_weights)
        # print("server_weights_pca: ", server_weights_pca)
        print("shape of server_weights_pca: ", server_weights_pca.shape)

        return clients_weights_pca, server_weights_pca


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
        self.pca_weights_clientserver_init = np.vstack((clients_weights_pca, server_weights_pca))
        print("shape of self.pca_weights_clientserver_init: ", self.pca_weights_clientserver_init.shape)
   
        # save a copy for later update in DQN training episodes
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy() 


    def add_client(self):
        # Add a new client to the server
        raise NotImplementedError
