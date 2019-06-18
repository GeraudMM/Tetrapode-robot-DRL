import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from torch.utils.data import Dataset, TensorDataset, DataLoader
from model import PPO_Actor_Critic

import matplotlib.pyplot as plt
import time


def is_nan(x):
    return (x is np.nan or x != x)

class PPO_Agent:
    def __init__(self, env, lr=0.0001, beta=0, learning_time=5, eps=0.2, tau=0.95, batch_size=128, constraint = 1.0):
        
        #Initialize environment
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        state_size = states.shape[1]
        self.env = env
        random_seed = random.randint(1, 100)
        
        #Initialize some hyper parameters of agent
        self.lr = lr
        self.learning_time = learning_time
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = 0.99
        self.batch_size = batch_size
        self.beta = beta#parameter for entropy panelty
        self.eps = eps#parameter for clip
        self.tau = tau#parameter for GAE
        
        #Networks and optimizers
        self.seed = random.seed(random_seed)
        self.network = PPO_Actor_Critic(state_size, action_size, random_seed, fc1_units=1024, fc2_units=1024)
        self.optimizer = optim.Adam(self.network.parameters(),lr=self.lr, eps=1e-8, weight_decay=1e-4)
        
        self.best = -100 # This saves the best average score over 10 episodes in past agents.
        
        self.constraint = constraint
        
    def act(self, states):
        self.network.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float)
            actions, log_probs, _, v = self.network(states)
            actions = actions.detach().cpu().numpy()
            log_probs = log_probs.detach().cpu().numpy()
            v = v.detach().cpu().numpy()
        self.network.train()
        return actions, log_probs, v
    
    def check(self):
        brain_name = self.env.brain_names[0]
        self.network.eval()
        total_rewards = 0
        reward_flag = False
        batch_check = 5
        for _ in range(batch_check):
            scores = np.zeros(self.num_agents)
            env_info = self.env.reset(train_mode=True)[brain_name]
            for t in range(1500):
                states = env_info.vector_observations
                action, _, _ = self.act(states)
                env_info = self.env.step(action)[brain_name]
                rewards = np.array(env_info.rewards)
                if any(np.isnan(rewards.reshape(-1))):
                    rewards[np.isnan(rewards)] = -5
                    reward_flag = True
                scores += rewards
            total_rewards += np.mean(scores)
        if reward_flag:
            print('\r NaN in rewards during testing!')
        self.network.train()
        self.network.cpu()
        if total_rewards/batch_check > self.best:
            torch.save(self.network.state_dict(), 'MYTetrapodeV2_Checkpoint.pth')
            self.best = total_rewards/batch_check
            print('\rCurrent network average: {:.1f}. Network Updated. Current best score {:.1f}'.format(total_rewards/batch_check, self.best))
        else:
            self.network.load_state_dict(torch.load('MYTetrapodeV2_Checkpoint.pth'))
            print('\rCurrent network average: {:.1f}. Network Reload. Current best score {:.1f}'.format(total_rewards/batch_check, self.best))
        self.network  
        
    
    def learn(self, states, actions, log_probs, dones, Advantages, Returns):
        '''
        This functions calculates the clipped surrogate function and do one step update to the action network
        And then the critic network will be updated
        The inputs are all lists of tensors. They are already sent to device before into the list
        '''
        
        #Generate dataset for getting small batches
        mydata = TensorDataset(states, actions, log_probs, dones, Advantages, Returns)
        Loader = DataLoader(mydata, batch_size = min(self.batch_size, len(mydata)//64), shuffle = True)
        
        for i in range(self.learning_time):
            for sampled_states, sampled_actions, sampled_log_probs, sampled_dones, sampled_advantages, sampled_returns in iter(Loader):
                _, new_log_probs, entropy, V = self.network(sampled_states, sampled_actions)
                ratio = (new_log_probs - sampled_log_probs).exp()
                
                Actor_Loss = -torch.min(input=ratio*sampled_advantages, other=torch.clamp(ratio, 1-self.eps, 1+self.eps)*sampled_advantages).mean()
                Entropy_Loss = -self.beta * entropy.mean()
                Critic_Loss = 0.5*(V-sampled_returns).pow(2).mean()
                Loss = Actor_Loss+Critic_Loss+Entropy_Loss
                self.optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                self.optimizer.step()
            
    def train(self, n_episode, max_t=1500, standardlize='row', method='MC', load=False):
        '''
        This function do the training part of the agent. The procedure is like:
            1. initialize environment
            2. Go through a whole episode, recode all states, actions, log_probs, rewards and dones information
            3. call learn function to update the networks
            4. repeat 2-3 for n_episode times.
        '''
        
        if load:
            self.network.cpu()
            self.network.load_state_dict(torch.load('MYTetrapodeV2_Checkpoint.pth'))
            self.network
            
        all_rewards = []
        total_window = deque(maxlen=100)
        brain_name = self.env.brain_names[0]
        score_window_moy = deque(maxlen=100)
        score_window_min = []
        score_window_max = []
        
        method = method.upper()
        if method not in ['MC', 'TD']:
            print('\rmethod can be only TD or MC. Input not supported! Use TD by default')
            method = 'TD'
        standardlize = standardlize.lower()
        if standardlize not in ['row', 'whole', 'none']:
            print('\rStandarlization in row or as a whole or none. Input not supported. Use row instead')
            standardlize = 'row'
            
        #self.check()
            
        states_history = []
        actions_history = []
        rewards_history = []
        log_probs_history = []
        dones_history = []
        values_history = []
        
        saves = [False for i in range(80)]
        start_time = time.time()
        nb_episode = 0
        
        for i in range(1, n_episode+1):
            #initialize environment
            env_info = self.env.reset(train_mode=True)[brain_name]
            
            total = np.zeros(self.num_agents)#Saves every reward signal
            scores_recorder = []#Saves the total reward whenever an agent is 'done'
            episodic_scores = np.zeros(self.num_agents)#Saves the current cumulated reward. Reset to 0 when agent is 'done'
            
            states = env_info.vector_observations
            actions, log_probs, v = self.act(states)
            
            reward_flag = False
            
            for _ in range(max_t):
                states_history.append(torch.tensor(states, dtype=torch.float))
                actions_history.append(torch.tensor(actions, dtype=torch.float))
                values_history.append(torch.tensor(v, dtype=torch.float))
                log_probs_history.append(torch.tensor(log_probs, dtype=torch.float))#Save as columns
                if any(np.isnan(actions.reshape(-1))):
                    print('\rCurrent episode {}. NaN in action!'.format(i))
                    self.network.cpu()
                    torch.save(self.network.state_dict(), 'MYTetrapode_Checkpoint_NaN_Action.pth')
                    return None
                env_info = self.env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                dones = torch.tensor(env_info.local_done, dtype=torch.float).view(-1,1)
                rewards = np.array(env_info.rewards)
                if any(np.isnan(rewards.reshape(-1))):
                    rewards[np.isnan(rewards)] = -5
                    reward_flag = True
                #Save reward info before turned into tensor
                total += rewards
                episodic_scores += rewards 
                
                rewards = torch.tensor(rewards, dtype=torch.float).view(-1,1)
                rewards_history.append(rewards)
                dones_history.append(dones)
                
                states = next_states
                actions, log_probs, v = self.act(states)
                
                for k in range(self.num_agents):
                    if env_info.local_done[k]:
                        scores_recorder.append(episodic_scores[k])
                        episodic_scores[k] = 0
                        
            if reward_flag:
                print('\rCurrent Episode {}! NaN in rewards!'.format(i))
                
                
            
            scores_recorder = np.array(scores_recorder)
            states_history.append(torch.tensor(states, dtype=torch.float))
            values_history.append(torch.tensor(v, dtype=torch.float))
            
            Advantages = []
            advantage = 0
            Returns = []
            returns = 0
            #Calculate advantages
            for j in reversed(range(len(states_history)-1)):
                if method == 'MC':
                    returns = rewards_history[j] + (1-dones_history[j])*returns*self.gamma
                else:
                    returns = rewards_history[j] + (1-dones_history[j])*values_history[j+1].detach()*self.gamma
                Returns.append(returns.view(-1))
                delta = rewards_history[j] + (1-dones_history[j])*self.gamma*values_history[j+1].detach() - values_history[j].detach()
                advantage = advantage*self.gamma*self.tau*(1-dones_history[j]) + delta
                Advantages.append(advantage.view(-1))
            Advantages.reverse()
            Advantages = torch.stack(Advantages).detach()
            if standardlize == 'row':
                Advantages = (Advantages - Advantages.mean(dim=1 ,keepdim=True))/Advantages.std(dim=1, keepdim=True)
            elif standardlize == 'whole':
                Advantages = (Advantages - Advantages.mean())/Advantages.std()
            Advantages = Advantages.view(-1,1)
            Returns.reverse()
            Returns = torch.stack(Returns).detach()
            Returns = Returns.view(-1,1)
            
            states_history = torch.cat(states_history[:-1], 0)
            actions_history = torch.cat(actions_history, 0)
            log_probs_history = torch.cat(log_probs_history, 0)
            dones_history = torch.cat(dones_history, 0)
            
            self.learn(states_history, actions_history, log_probs_history, dones_history, Advantages, Returns)
            
            states_history = []
            actions_history = []
            rewards_history = []
            log_probs_history = []
            dones_history = []
            values_history = []
            
            score_window_moy.append(np.nanmean(scores_recorder))
            score_window_min.append(np.min(scores_recorder))
            score_window_max.append(np.max(scores_recorder))
            total_window.append(np.mean(total))
            all_rewards.append(np.nanmean(scores_recorder))
            now_time = time.time() - start_time
            nb_episode +=1
            
            print('\rEpisode {}. Total score {:.1f}, average score {:.1f}, past total average {:.1f}, past average {:.1f}, best {:.1f}, worst {:.1f}, time: {:.0f}s'.format(i, np.mean(total), np.mean(scores_recorder), np.mean(total_window),np.mean(score_window_moy), np.max(scores_recorder), np.min(scores_recorder), now_time, end=""))
            
            if i % 20 == 0:
                fig = plt.figure(figsize=(20,10))
                ax = fig.add_subplot(111)
                plt.plot(all_rewards,label = 'Moy scores', linewidth=4)
                #plt.plot(np.mean(all_rewards,axis=1),label = 'Mean of the episodes', linewidth=4)
                plt.plot(score_window_max,label = 'Max Scores')
                plt.plot(score_window_min,label = 'Min Scores')
                plt.ylabel('Score')
                plt.xlabel('Episode #')
                plt.show()
                
                #self.check()
                
            for i_save in range(len(saves)):
                if np.mean(total)>(100 * (1 + i_save) ) and saves[i_save] == False:
                    saves[i_save] = True
                    torch.save(self.network.state_dict(),'MYTetrapodeV2_Checkpoint.pth')
                    print('\rSaved actor and local network for Total score is >= {:.2f}'.format(100 * (1 + i_save), end=""))
                    nb_episode = 0 
                    
            if nb_episode >=20:
                print('\rNetwork Reload!')
                self.network.cpu()
                self.network.load_state_dict(torch.load('MYTetrapodeV2_Checkpoint.pth'))
                self.network
                nb_episode = 0 
                
                
        np.save('PPO_rewards.npy',np.array(all_rewards))
        return all_rewards
    
    def test(self, t_step, standardlize='row'):

        brain_name = self.env.brain_names[0]
        
        standardlize = standardlize.lower()
        if standardlize not in ['row','whole']:
            print('\rStandarlization in row or as a whole. Input not supported. Use row instead')
            standardlize = 'row'
        
        #initialize environment
        all_step_rewards = []
        env_info = self.env.reset(train_mode=False)[brain_name]
        
        states = env_info.vector_observations
        actions, log_probs, v = self.act(states)
        
        for n_step in range(t_step):
            env_info = self.env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            dones = torch.tensor(env_info.local_done, dtype=torch.float).view(-1,1)
            rewards = torch.tensor(env_info.rewards, dtype=torch.float).view(-1,1)
            
            states = next_states
            actions, log_probs, v = self.act(states)                                  
            all_step_rewards.append(np.mean(env_info.rewards))
        
        print('\r score {:.1f}'.format(np.sum(all_step_rewards),end=''))               
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        plt.plot(all_step_rewards,label = 'Reward per Step')
        plt.ylabel('Reward')
        plt.xlabel('Step #')
        plt.show()
       
        return all_step_rewards