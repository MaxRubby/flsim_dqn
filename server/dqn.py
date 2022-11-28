import logging
import random
from server import Server
from sklearn.cluster import KMeans
from threading import Thread
import utils.dists as dists  # pylint: disable=no-name-in-module

from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
from lib import plotting

class DQNServer(Server):
    """Federated learning server that performs KMeans profiling during selection."""

    def __init__(self,config,case_name):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=options.replay_memory_size)
        self.total_steps = 0
        super().__init__(config,case_name)

    def _build_model(self):
        layers = self.options.layers

        states = Input(shape=self.env.observation_space.shape)
        z = states
        for l in layers:
            z = Dense(l, activation='relu')(z)

        q = Dense(self.env.action_space.n, activation='linear')(z)

        model = Model(inputs=[states], outputs=[q])
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=huber_loss)

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy(self, state):

        nA = 100
        epsilon = 0.5
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(self.model.predict([[state]])[0])
        action_probs[best_action] += (1 - epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def create_greedy_policy(self):

        def policy_fn(state):
            return np.argmax(self.model.predict([[state]])[0])

        return policy_fn

    def train_episode(self):

        # Reset the environment
        state = self.model_weights(self.clients)
        terminal_state = False
        while terminal_state == False:
            action = self.epsilon_greedy(state)
            next_state, reward, done, _ = self.step(action)
            self.memorize(state, action, reward, next_state, done)
            self.replay()
            terminal_state = done
            state = next_state

    def replay(self):
        batch_size = 10
        if len(self.memory) > batch_size:
            sample_batch = random.sample(self.memory, self.options.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in sample_batch:
                states.append(state)

                if done == False:
                    max_next_value = np.max(self.target_model.predict([[next_state]])[0])
                    new_q_value = reward + self.options.gamma * max_next_value
                else:
                    new_q_value = reward
                    # Update Q value for given state
                q = self.model.predict([[state]])[0]
                q[action] = new_q_value
                target_q.append(q)
            states = np.array(states)
            target_q = np.array(target_q)
            self.model.fit(states, target_q, epochs=1, verbose=0)

            if done:
                self.total_steps += 1

            # If counter reaches set value, update target network with weights of main network
            if self.total_steps > self.options.update_target_estimator_every:
                self.update_target_model()
                self.total_steps = 0

    # Run federated learning
    def run(self):
        # Perform profiling on all clients
        # self.profile_clients()

        # Continue federated learning
        super().run()

    # Federated learning phases
    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round
        cluster_labels = self.clients.keys()

        # Generate uniform distribution for selecting clients
        dist = dists.uniform(clients_per_round, len(cluster_labels))

        # Select clients from KMeans clusters
        sample_clients = []
        for i, cluster in enumerate(cluster_labels):
            # Select clients according to distribution
            if len(self.clients[cluster]) >= dist[i]:
                k = dist[i]
            else:  # If not enough clients in cluster, use all avaliable
                k = len(self.clients[cluster])

            sample_clients.extend(random.sample( # random sample
                self.clients[cluster], k))

         # Shuffle selected sample clients
        random.shuffle(sample_clients)

        return sample_clients

    def step(self):
        reward = self.round() ## accuracy
        next_state = self.model_weights(self.clients)
        done = self.memory.length ==  self.config.fl.rounds-1
        return reward,next_state,done

    # Output model weights
    def model_weights(self, clients):
        # Configure clients to train on local data
        self.configuration(clients)

        # Train on local data for profiling purposes
        threads = [Thread(target=client.train) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client reports
        reports = self.reporting(clients)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        return [self.flatten_weights(weight) for weight in weights]

    def prefs_to_weights(self):
        prefs = [client.pref for client in self.clients]
        return list(zip(prefs, self.model_weights(self.clients)))

    def profiling(self, clients):
        # Perform clustering

        weight_vecs = self.model_weights(clients)

        # Use the number of clusters as there are labels
        n_clusters = len(self.loader.labels)

        logging.info('KMeans: {} clients, {} clusters'.format(
            len(weight_vecs), n_clusters))
        kmeans = KMeans(  # Use KMeans clustering algorithm
            n_clusters=n_clusters).fit(weight_vecs)

        return kmeans.labels_

    # Server operations
    def profile_clients(self):
        # Perform profiling on all clients
        kmeans = self.profiling(self.clients)

        # Group clients by profile
        grouped_clients = {cluster: [] for cluster in
                           range(len(self.loader.labels))}
        for i, client in enumerate(self.clients):
            grouped_clients[kmeans[i]].append(client)

        self.clients = grouped_clients  # Replace linear client list with dict

    def add_client(self):
        # Add a new client to the server
        raise NotImplementedError
