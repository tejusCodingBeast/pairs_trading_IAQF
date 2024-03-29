"""
TraderEnv class
author: Manav Shah
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class TraderEnv(gym.Env):

    def __init__(self,p1,p2,trainIters=100,batchNumber=0,negRewM=1.0,startTrainIdx=22,sd=60):
        #trading param

        self.trainMax = trainIters #should match TRAINING_ITERS in learning algorithm
        self.rnd = 2
        self.pairs=[[p1,p2]]  
        self.lags = 22
        self.dataidx = self.lags
        self.df = pd.read_csv("etfs.csv")
        self.df = self.df.iloc[:2518,:].copy() # Get rid of data after 2019-12-31
        self.df = self.df[["Date",p1,p2]].copy() # trim all data except pair data
        self.completeData = ""+self.pairs[0][0]+"."+self.pairs[0][1]+".full.csv"
        print("completeData: ",self.completeData)
        self.dflen = len(self.df)
        self.training = True
        self.trainIters = 0
        self.testIdx = 1764 #idx in dataset where test data begins: 2017-01-01
        self.startTrainIdx = startTrainIdx 
        self.negRewMult = negRewM
        self.maxIdx = 10000000
        self.firstEp = True
        self.cumTestR = 0.0
        self.verified = False
        self.cumTestRs = []
        self.prc1 = []
        self.prc2 = []
        self.trainSps = []
        self.testSp = []
        self.testActs = []
        self.testActCt = []
        self.testMap = [[],[],[],[],[],[],[],[],[],[]]
        #env params
        self.action_space = spaces.Discrete(3)# [-1,0,1] #actions are: -1 short, 0 no position, 1 long
        self.high = np.array([3, 3, 1, 0, 0, 1, 1],dtype='float32')
        
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)
        self.batchNumber = batchNumber
        self.seed(sd)
        
        
        #building environment states for each pair
        if not os.path.isfile(self.completeData):
            print("CREATING COMPLETE DATA")
            j = 0
            for p in self.pairs:
                
                self.df["sp"+str(j)] = np.abs( self.df[p[0]] - self.df[p[1]] )
                
                self.df["sprets"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #22 day moving average
                self.df["spmn"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #19 mva
                self.df["spmn_3"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn_3"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #17 mva
                self.df["spmn_5"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn_5"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #12 mva
                self.df["spmn_10"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn_10"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #7 mva
                self.df["spmn_15"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn_15"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                #5 mva
                self.df["spmn_17"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                self.df["sp_spmn_17"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                
                self.df["spmn_5tau"+str(j)] = pd.Series([0 for x in range(len(self.df["sp"+str(j)]))])
                
                self.df["sp_stdev"+str(j)] = pd.Series([0.0 for x in range(len(self.df["sp"+str(j)]))])
                
                #TODO: try higher precision round, 4 not 2
                #calc spread returns
                for i in range(len(self.df["sp"+str(j)])-1): 
                    self.df["sprets"+str(j)][i+1] = np.round( ((self.df["sp"+str(j)][i+1] - self.df["sp"+str(j)][i])/self.df["sp"+str(j)][i]),self.rnd)
                
                # sp/spmn22 (lags)
                #going back to spread mean and spread/spreadMean for now
                for i in range(len(self.df["sp"+str(j)])-self.lags): 
                    self.df["spmn"+str(j)][self.lags+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+self.lags])), self.rnd)
                
                self.df["sp_spmn"+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn"+str(j)],self.rnd)
                
                # sp/spmn19 (lags-3)
                #going back to spread mean and spread/spreadMean for now
                l = 3
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    self.df["spmn_"+str(l)+str(j)][(self.lags-l)+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+(self.lags-l)])), self.rnd)
                
                self.df["sp_spmn_"+str(l)+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn_"+str(l)+str(j)],self.rnd)
                
                # sp/spmn17 (lags-5)
                #going back to spread mean and spread/spreadMean for now
                l = 5
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    self.df["spmn_"+str(l)+str(j)][(self.lags-l)+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+(self.lags-l)])), self.rnd)
                
                self.df["sp_spmn_"+str(l)+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn_"+str(l)+str(j)],self.rnd)
                
                # sp/spmn12 (lags-10)
                #going back to spread mean and spread/spreadMean for now
                l = 10
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    self.df["spmn_"+str(l)+str(j)][(self.lags-l)+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+(self.lags-l)])), self.rnd)
                
                self.df["sp_spmn_"+str(l)+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn_"+str(l)+str(j)],self.rnd)
                
                # sp/spmn7 (lags-15)
                #going back to spread mean and spread/spreadMean for now
                l = 15
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    self.df["spmn_"+str(l)+str(j)][(self.lags-l)+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+(self.lags-l)])), self.rnd)
                
                self.df["sp_spmn_"+str(l)+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn_"+str(l)+str(j)],self.rnd)
                
                # sp/spmn5 (lags-17)
                #going back to spread mean and spread/spreadMean for now
                l = 17
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    self.df["spmn_"+str(l)+str(j)][(self.lags-l)+i] = np.round( (np.mean(self.df["sp"+str(j)][i:i+(self.lags-l)])), self.rnd)
                
                self.df["sp_spmn_"+str(l)+str(j)] = np.round(self.df["sp"+str(j)] / self.df["spmn_"+str(l)+str(j)],self.rnd)
                
                l=5
                for i in range(len(self.df["sp"+str(j)])-(self.lags-l)): 
                    if self.df["sp_spmn_"+str(l)+str(j)][(self.lags-l)+i] > 1.04 and self.df["spmn_"+str(l)+"tau"+str(j)][((self.lags-l)+i)-1] != -1 : 
                        self.df["spmn_"+str(l)+"tau"+str(j)][(self.lags-l)+i] = -1
                    elif self.df["sp_spmn_"+str(l)+str(j)][(self.lags-l)+i] < 0.96 and self.df["spmn_"+str(l)+"tau"+str(j)][((self.lags-l)+i)-1] != 1: 
                        self.df["spmn_"+str(l)+"tau"+str(j)][(self.lags-l)+i] = 1
                    else:
                        self.df["spmn_"+str(l)+"tau"+str(j)][(self.lags-l)+i] = self.df["spmn_"+str(l)+"tau"+str(j)][((self.lags-l)+i)-1]
                
                
                '''
                #calc srets mean
                for i in range(len(self.df["sp"+str(j)])-self.lags): 
                    self.df["spmn"+str(j)][self.lags+i] = np.round( (np.mean(self.df["sprets"+str(j)][i:i+self.lags])), self.rnd)
                
                self.df["sp_spmn"+str(j)] = np.round(self.df["sprets"+str(j)] / self.df["spmn"+str(j)],self.rnd)
                '''
                
                # print("dataframe: ",self.df[[p[0],p[1],"sp"+str(j),"sprets"+str(j),"spmn"+str(j),"sp_spmn"+str(j),\
                # "spmn_3"+str(j),"sp_spmn_3"+str(j),"spmn_5"+str(j),"sp_spmn_5"+str(j),\
                # "spmn_10"+str(j),"sp_spmn_10"+str(j),"spmn_15"+str(j),"sp_spmn_15"+str(j),"spmn_17"+str(j),"sp_spmn_17"+str(j),"spmn_5tau0"]])
                #input()
                j += 1
            # self.df.to_csv(self.completeData)
            
        else:#we have data
            p = self.pairs[0]
            j = 0
            # print("LOADING COMPLETE DATA")
            self.df = pd.read_csv(self.completeData)
            # print("dataframe: ",self.df[[p[0],p[1],"sp"+str(j),"sprets"+str(j),"spmn"+str(j),"sp_spmn"+str(j),"spmn_3"+str(j),"sp_spmn_3"+str(j),"spmn_5"+str(j),"sp_spmn_5"+str(j),"spmn_5tau0"]])
            #input()
                
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        self.dataidx += 1
        idx = self.dataidx
        
        sp0 = self.df["sp0"][self.dataidx]
        sprets0 = self.df["sprets0"][self.dataidx]
        spmn0 = self.df["spmn0"][self.dataidx]
        sp_spmn0 = self.df["sp_spmn0"][self.dataidx]
        spmn_30 = self.df["spmn_30"][self.dataidx]
        sp_spmn_30 = self.df["sp_spmn_30"][self.dataidx]
        spmn_50 = self.df["spmn_50"][self.dataidx]
        sp_spmn_50 = self.df["sp_spmn_50"][self.dataidx]
        
        spmn_100 = self.df["spmn_100"][self.dataidx]
        sp_spmn_100 = self.df["sp_spmn_100"][self.dataidx]
        spmn_150 = self.df["spmn_150"][self.dataidx]
        sp_spmn_150 = self.df["sp_spmn_150"][self.dataidx]
        spmn_170 = self.df["spmn_170"][self.dataidx]
        sp_spmn_170 = self.df["sp_spmn_170"][self.dataidx]
        
        spmn_5tau0 = self.df["spmn_5tau0"][self.dataidx]
        
        # sprets1 = self.df["sprets1"][self.dataidx]
        # sp_spmn1 = self.df["sp_spmn1"][self.dataidx]
        
        # print("sprets: ",sprets)
        # print("sp_spmn: ",sp_spmn)
        #signal = -1 if action == 0 else 1 #calculate signal for 0 or 1
        signal = action - 1 #converting actions 0,1,2 to signals -1,0,1
        rewards = sprets0 * signal#action comes as index 0,1,2 converting to position -1,0,1
        if self.training and rewards < 0.0: 
            rewards *= self.negRewMult # making negative returns much worse when training
        
        #if self.trainIters==(self.trainMax-1): print("sprets0, rewards: ",sprets0,rewards)
        if self.dataidx >= self.testIdx: 
            #verifying NN
            if self.dataidx == (((self.dflen - self.testIdx)/2) + self.testIdx):#halfway through test data
                # print("verifying data",self.dataidx)
                #input()
                if self.cumTestR > 0.0:
                    self.verified = True
            
            #print("new action: ",action)
            #print("self.dataidx, self.testIdx: ",self.dataidx,self.testIdx)
            '''
            print("spmn_5tau0,sp0,spmn0,sp_spmn0,sprets0,signal,spmn_30,"\
            +"sp_spmn_30,spmn_50,sp_spmn_50: ", \
            spmn_5tau0,sp0,spmn0,sp_spmn0,sprets0,signal,spmn_30,\
            sp_spmn_30,spmn_50,sp_spmn_50)
            '''
            self.cumTestR += rewards
            self.cumTestRs.append(self.cumTestR)
            self.prc1.append(self.df[self.pairs[0][0]][self.dataidx])
            self.prc2.append(self.df[self.pairs[0][1]][self.dataidx])
            self.testSp.append(self.df["sp0"][self.dataidx])
            self.testActs.append(signal)
            
            #self.testMap[0].append(spmn_5tau0)#0
            self.testMap[0].append(sp0)#1
            self.testMap[1].append(spmn0)#2
            self.testMap[2].append(sp_spmn0)#3
            self.testMap[3].append(sprets0)#4
            #self.testMap[5].append(signal)#5
            self.testMap[4].append(spmn_30)#6
            self.testMap[5].append(sp_spmn_30)#7
            self.testMap[6].append(spmn_50)#8
            self.testMap[7].append(sp_spmn_50)#9
            self.testMap[8].append(spmn_100)#10
            self.testMap[9].append(sp_spmn_100)#11
            #self.testMap[12].append(spmn_150)
            #self.testMap[13].append(sp_spmn_150)
            #self.testMap[14].append(spmn_170)
            #self.testMap[15].append(sp_spmn_170)
            
            #10 features sp0,sprets0,
            print("rew,sprets,sig,act,cumTestR,verified: ",rewards,sprets0,signal,action,self.cumTestR,self.verified)
            
        # print("self.dataidx: ",self.dataidx)
        #return np.array((3.0,3.0,3.0,3.0)),1.0,False,{}
        
        # 
        done = False
        if self.training:
            done = self.dataidx == self.testIdx-1 #finished training, before we reach test data
        else:
            done = (self.dataidx == self.dflen-1 or self.dataidx == self.maxIdx)#finished testing at end of data set
            
        if done and self.training: 
            self.trainIters += 1
            # print("self.trainIters: ",self.trainIters)
        if self.trainIters == self.trainMax: 
            self.training = False
            
        #record training data once
        if self.firstEp:
            self.trainSps.append(sp0)
            
        if done and self.firstEp:
            self.firstEp = False
            # plt.plot(self.trainSps)
            # plt.savefig(str(self.batchNumber)+"/"+str(self.pairs[0][0])+"_"+str(self.pairs[0][1])+".trainData."+str(self.trainMax)+".png")
            # plt.show()
            # plt.close()
            
            #reset matplotlib for testing data subplots
            #matplotlib
            self.f, self.axarr = plt.subplots(5, sharex=True, figsize=(7,10))
            self.f.subplots_adjust(hspace=0.22)
            
        #done training and testing printing results
        if done and not self.training and self.dataidx > self.testIdx:
            # print(self.dataidx,self.dflen)
            
            self.testActCt.append(self.testActs.count(-1))
            self.testActCt.append(self.testActs.count(0))
            self.testActCt.append(self.testActs.count(1))
            # print("self.testActCt: ",self.testActCt)
            self.createMap()
            self.axarr[0].set_title("Cummulative Returns")
            self.axarr[0].plot(self.cumTestRs)#,colors=test_rcolors)
            self.axarr[1].plot(self.prc1)
            self.axarr[1].plot(self.prc2)
            self.axarr[1].set_title(str(self.pairs[0][0])+"/"+str(self.pairs[0][1])+" Prices")
            self.axarr[1].legend()
            self.axarr[2].plot(self.testSp)
            self.axarr[2].set_title(str(self.pairs[0][0])+"/"+str(self.pairs[0][1])+" Spread")
            # print("self.testMap: ",self.testMap)
            self.axarr[3].imshow(np.array(self.testMap),aspect="auto",cmap="jet")
            self.axarr[3].set_yticklabels(["spread","spreadReturns","spreadMean5day","sp/spreadMean5day","spreadMean7day","sp/spreadMean7day","spreadMean10day","sp/spreadMean10","spreadMean15day","sp/spreadMean15"])
            self.axarr[3].tick_params(axis='y', which='major', labelsize=6)
            self.axarr[3].set_title("Test Data Features")
            #self.axarr[3].figure(figsize=(100,100))
            self.axarr[4].plot(self.testActs)
            self.axarr[4].set_title("Test Data Actions")
            self.axarr[4].set_xlabel("Test Data Trading Days",fontsize=15)
            # plt.savefig(str(self.batchNumber)+"/"+str(self.pairs[0][0])+"_"+str(self.pairs[0][1])+".test_r.eps"+str(self.trainMax)+".png")
            # plt.show()
            # plt.close()
            
            self.mtplt()
            
            
            #print("heatMap: ",self.testMap)
            
        #fix this, add 100, 150, 170
        '''
        return np.array((spmn_5tau0,sp0,spmn0,sp_spmn0,sprets0,signal,spmn_30,\
                sp_spmn_30,spmn_50,sp_spmn_50,spmn_100,sp_spmn_100,spmn_150,sp_spmn_150,\
            spmn_170,sp_spmn_170),dtype='float32'), rewards, done, {}#sp_spmn0 to first 0.0
        '''
        return np.array((sprets0,sp_spmn0,sp_spmn_30,sp_spmn_50,sp_spmn_100,sp_spmn_150,sp_spmn_170),dtype='float32'), rewards, done, {} #7 features
        
    def mtplt(self):
        
        # plt.bar(["short","no position","long"],self.testActCt, align='center', alpha=0.5)
        # #plt.figure(figsize=(18, 10)).suptitle("mytitle",fontsize=20)
        # plt.title("Number of Actions")
        # plt.savefig(str(self.batchNumber)+"/"+str(self.pairs[0][0])+"_"+str(self.pairs[0][1])+".test_numActs.eps"+str(self.trainMax)+".png")
        # plt.show()
        # plt.close()
        pass
        
    def reset(self):
        #lags = self.lags#start at beginning
        lags = self.startTrainIdx #start at specific point in training data
        if self.training:
            #self.dataidx = self.lags # go through full training set each time
            #step up dataidx, meaning current data is given higher weight, similar to exponentially weighted moving average
            self.dataidx = lags + int((self.testIdx-1-lags)/self.trainMax) * self.trainIters
        else:
            self.dataidx = self.testIdx
        
        #print(self.dataidx,self.lags)
        
        sp0 = self.df["sp0"][self.dataidx]
        sprets0 = self.df["sprets0"][self.dataidx]
        signal = 0
        spmn0 = self.df["spmn0"][self.dataidx]
        sp_spmn0 = self.df["sp_spmn0"][self.dataidx]
        spmn_30 = self.df["spmn_30"][self.dataidx]
        sp_spmn_30 = self.df["sp_spmn_30"][self.dataidx]
        spmn_50 = self.df["spmn_50"][self.dataidx]
        sp_spmn_50 = self.df["sp_spmn_50"][self.dataidx]
        
        spmn_100 = self.df["spmn_100"][self.dataidx]
        sp_spmn_100 = self.df["sp_spmn_100"][self.dataidx]
        spmn_150 = self.df["spmn_150"][self.dataidx]
        sp_spmn_150 = self.df["sp_spmn_150"][self.dataidx]
        spmn_170 = self.df["spmn_170"][self.dataidx]
        sp_spmn_170 = self.df["sp_spmn_170"][self.dataidx]
        
        spmn_5tau0 = self.df["spmn_5tau0"][self.dataidx]
        #print(sp0,sprets0,spmn0,sp_spmn0)
        '''
        
        res = np.array((spmn_5tau0,sp0,spmn0,sp_spmn0,sprets0,signal,spmn_30,sp_spmn_30,\
        spmn_50,sp_spmn_50,spmn_100,sp_spmn_100,spmn_150,sp_spmn_150,spmn_170,sp_spmn_170),dtype='float32')#sp0,spmn0
        '''
        
        res = np.array((sprets0,sp_spmn0,sp_spmn_30,sp_spmn_50,sp_spmn_100,sp_spmn_150,sp_spmn_170),dtype='float32') #7 features
        
        #print("resetting: ",res)
        
        return res
            
    def createMap(self):
        tm = self.testMap
        newtm = self.testMap.copy()
        
        for i in range(len(tm)):
            for j in range(len(tm[i])):
                newtm[i][j] = (tm[i][j] - min(tm[i])) / (max(tm[i]) - min(tm[i]))
        
        self.testMap = newtm
        
        '''
        print("newtm",newtm)
        input()
        print("tm",tm)
        input()
        '''