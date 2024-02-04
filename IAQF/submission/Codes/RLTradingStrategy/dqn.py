"""
dqn.py
author: Manav Shah
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trader import * #trader.py
import json
import os


SEED = 68
torch.manual_seed(SEED)
np.random.seed(SEED)
BATCH_NUMBER = 134
TRAINING_ITERS = 30
HIDDEN_LAYER = 50


NEG_REWM = 1.0 #1000.0 
START_TRAINIDX = 254 #2011-01-01

# Hyper Parameters
BATCH_SIZE = 32 
LR = 0.01                   # learning rate
EPSILON = 0.9              # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 200

N_ACTIONS = N_STATES = ENV_A_SHAPE = 0.0

# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

def calculate_metrics(cumret):

    rets = pd.DataFrame(cumret).pct_change()
    sharpe = np.sqrt(252) * np.nanmean(rets) / np.nanstd(rets)

    # maxdd and maxddd
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=cumret[t]/highwatermark[t]-1
        
    maxDD=np.min(drawdown)
    
    return sharpe, maxDD


print("N_ACTIONS, N_STATES, ENV_A_SHAPE: ",N_ACTIONS, N_STATES, ENV_A_SHAPE)
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        #print("N_STATES, N_ACTIONS, ENV_A_SHAPE, HIDDEN_LAYER: ", N_STATES, N_ACTIONS, ENV_A_SHAPE, HIDDEN_LAYER)
        #input()
        self.fc1 = nn.Linear(N_STATES, HIDDEN_LAYER)#was 50
        self.fc1.weight.data.normal_(0, 0.1)#0.1   # initialization
        self.out = nn.Linear(HIDDEN_LAYER, N_ACTIONS)#was 50
        self.out.weight.data.normal_(0, 0.1)#0.1   # initialization
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)#F.relu
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2),dtype='float32')     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.actions = []
        self.training = True
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        #print("x: ",x)
        
        
        # input only one sample
        if np.random.uniform() < EPSILON or (not self.training):   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            #self.actions.append(1)
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            #self.actions.append(0)
            # if not self.training:
            #     # print("action: ",action)
            #     # input()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

         # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def clearReplayMemory(self):
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2),dtype='float32')
        self.memory_counter = 0                                         # for storing memory
        


import json
# Writing to sample.json
with open("cluster_models.json", "r") as openfile:
    cluster_analysis_pairdict = json.load(openfile)

# with open("stat_pairs.json", "r") as openfile:
#     cluster_analysis_pairdict = json.load(openfile)


pairs = cluster_analysis_pairdict['k-means'][0] + cluster_analysis_pairdict['k-means'][1] # + cluster_analysis_pairdict['k-means'][2]

# pairs = [["ACWI","SPDW"]]

totalReturns = []
pairReturns = {}
totalSharpe = []
totalMaxDD = []

pairReturns["seed"] = SEED
pairReturns["trainIters"] = TRAINING_ITERS
pairReturns["negRewM"] = NEG_REWM
pairReturns["startTrainIdx"] = START_TRAINIDX
pairReturns["hiddenLayer"] = HIDDEN_LAYER
pairReturns["sharpe"] = {'avg':0}
pairReturns["maxDD"] = {'avg':0}


finalReturns = 0.0
for p in pairs:
    env = TraderEnv(p[0],p[1],TRAINING_ITERS,BATCH_NUMBER,NEG_REWM,START_TRAINIDX,SEED).unwrapped
    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

    dqn = DQN()
    
    print("Running ",TRAINING_ITERS," episodes...")
    cum_r = 0.0
    cum_rs = []
    testing = False
    
    for i_episode in range(TRAINING_ITERS+1):#
        s = env.reset()
        EPSILON = min(1.0,(i_episode / TRAINING_ITERS) * 1.1) # 2.0
        if(i_episode == (TRAINING_ITERS-1)): 
            print("----LAST TRAINING EPISODE, EPSILON=1.0----")
            EPSILON = 1.0 # last training episode, full greedy
        if(i_episode == TRAINING_ITERS): 
            dqn.training = False
            
            print("----TRAINING COMPLETE----")
            # print("----CLEARING REPLAY MEMORY PRIOR TO TESTING---")
            # dqn.clearReplayMemory()
            print("----BEGIN TESTING, EPSILON=1.0----")
            EPSILON = 1.0
            testing = True
            '''
            for name, param in dqn.eval_net.named_parameters():
                print(name, '\t\t', param.shape)
            for name, param in dqn.target_net.named_parameters():
                print(name, '\t\t', param.shape)
            print(dqn.eval_net.fc1.weight)
            print(dqn.target_net.fc1.weight)
            '''
            #Turn off training
            #model.train(False) so Dropout and Batchnorm are in test mode.
            dqn.eval_net.train(False)
            dqn.target_net.train(False)
            dqn.eval_net.eval()
            dqn.eval_net.eval()
            '''
            #testing momentum
            print("t.momentum,e.momentum: ",dqn.target_net.fc1.momentum, dqn.eval_net.fc1.momentum)
            input()
            dqn.target_net.fc1.momentum = 0
            dqn.target_net.out.momentum = 0#.momentum = 0 #0.1 is default
            dqn.eval_net.fc1.momentum = 0 #0.1 is default
            dqn.eval_net.out.momentum = 0
            '''
            #print(dqn.actions)
            # print("last training done, starting testing...")
            # input()
            
        #print("s: ",s)
        ep_r = 0
        while True:
            a = dqn.choose_action(s)
    
            # take action
            s_, r, done, info = env.step(a)

            # print(s_)
            #if testing: print("sars: ",s,a,r,s_)
    
            dqn.store_transition(s, a, r, s_)
            
            ep_r += r
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done and not testing:
                    print('Ep: ', i_episode,'| EPSILON: ', EPSILON,'| Ep_r: ', np.round(ep_r, 2))
                    cum_rs.append(ep_r)

                if done and testing:
                    
                    sharpe, maxDD = calculate_metrics(np.nancumprod(np.array(cum_rs)+1))

                    totalReturns.append(np.round(ep_r, 2))
                    totalSharpe.append(np.round(sharpe, 3))
                    if not math.isinf(maxDD):
                        totalMaxDD.append(np.round(maxDD,3))
                    pairReturns[p[0]+"_"+p[1]] = np.round(ep_r, 2)
                    pairReturns["totalReturns"] = totalReturns
                    pairReturns["AvgCumRets"] = np.round(np.mean(totalReturns),4)
                    pairReturns["sharpe"]['avg'] = np.round(np.mean(totalSharpe),4)
                    pairReturns["maxDD"]['avg'] = np.round(np.mean(totalMaxDD),4)
                    pairReturns["sharpe"]['avg'] = np.round(np.mean(totalSharpe),4)
                    pairReturns["maxDD"]['avg'] = np.round(np.mean(totalMaxDD),4)
                    pairReturns["sharpe"][p[0]+"_"+p[1]] = np.round(sharpe, 3)
                    pairReturns["maxDD"][p[0]+"_"+p[1]] = np.round(maxDD, 3)


                    #print("totalReturns: ",totalReturns)
                    print("pairReturns: ",pairReturns)
                    print("AvgCumRets: ",np.mean(totalReturns))
                    print("sharpe-avg",np.round(np.mean(totalSharpe),4))
                    print("maxDD-avg",np.round(np.mean(totalMaxDD),4))
                    
                    with open('cluster_pairReturns/pairReturns.json', 'w') as fp:
                        json.dump(pairReturns, fp)

                    # with open('stat_pairReturns/pairReturns.json', 'w') as fp:
                    #     json.dump(pairReturns, fp)
            if done:
                break
            s = s_
        #if not dqn.training: print(dqn.actions)
            
