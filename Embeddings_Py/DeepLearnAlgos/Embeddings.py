from Distributions.Zipfian import Zipfian
from threading import Lock
import numpy as np
import random
from math import log

class Embeddings(object):
    """description of class"""
    def __init__(self, v, n, eta, initial, embeddingsfile = "", contextsfile = ""):
        self.Eta = eta
        self.V = v # If the word index starts from 1 then 1 should be added to v before calling
        self.N = n # Embeddings dimension

        """
        self.hid_weights = np.empty((self.V, self.N), dtype = float)
        self.out_weights = np.empty((self.V, self.N), dtype = float)
        self.hid_weightslocks = []
        self.out_weightslocks = []
        print ("Creating lock objects")
        for i in range(self.V):
            self.hid_weightslocks.append(Lock())
            self.out_weightslocks.append(Lock())
        print ("Finished creating lock objects")
        """

        self.zipf = Zipfian(1.0, self.V - 1) # to ensure that a number between 0 and V is returned. V should
        # not be returned.
        if initial == True:
            print("Initializing weights")
            self.initializeWeights()
            print("Initialized weights")
        else:
            # Add code to load npy data from embeddings file and contexts file
            self.hid_weights = np.load(embeddingsfile)
            self.out_weights = np.load(contextsfile)

        self.embedsfile = embeddingsfile
        self.contextsfile = contextsfile
            
    """
        Another variation to initialize weights, if this doesnt work is
        to multiple randoffset. randoffset = np.sqrt(1/(self.V + self.N))
    """
    def initializeWeights(self):
        a = np.sqrt(1/(self.V + self.N)) 
        self.hid_weights = 2 * a * np.random.random_sample(size=(self.V, self.N)) - a
        self.hid_weights = self.hid_weights.astype('float32')
        self.out_weights = 2 * a * np.random.random_sample(size=(self.V, self.N)) - a
        self.out_weights = self.out_weights.astype('float32')

    def sigma(self,input):
        if input > 6:
            return 1
        elif input < -6:
            return 0
        else:
            return 1/(1 + np.exp(input * -1))

    """
        This method returns a list of sampleSize with word indices that 
        needs to be reinforced in the negative context. It will ensure that 
        the word indices are unique and are not part of the context words 
        that needs to be positively reinforced. This method assumes that a 
        mini-batch is being provided for training. contextwords is numpy 2d 
        array of int. If no mini-batch then the first dimension of the shape
        should be 1.
    """
    def generate_negative_samples(self, samplesize, contextwords):
        neg_samples = []
        i = 0
        while i < samplesize:
            wordindex = self.zipf.zip_invcdf(random.uniform(0,1))
            in_context = False
            for j in range(contextwords.shape[0]):
                for k in range(contextwords.shape[1]):
                    if contextwords[j,k] == wordindex:
                        in_context = True
            for j in range(i):
                if wordindex == neg_samples[j]:
                    in_context = True
            if not in_context:
                neg_samples.append(wordindex)
                i += 1
        return neg_samples

    '''
        The inputwordindex and the contextwordindex list will start from 1. 0 is reserved for 'UNK'
        The parameters for this method are
        1: batchSize - The number of training samples in this mini-batch
        2: inputWordIndex - Input word to be trained. 1D np array. The length denotes the batch size
        3: contextWordIndex - 2D np array of context word indices. Dimension 0 is batch and 1 is context words 
        4: negSamples - Number of negative samples to use for this batch. The negSamples chosen
                        will be ensured that they are not part of the context word Index
    ''' 
        
    def train_weights(self, batch_size, inputwordindex, contextwordindex, negsamplesize):
        # Ensure that the dimensions for inputWordIndex and contextWordIndex are the same as batchSize
        if inputwordindex.shape[0] != batch_size:
            raise Exception("The batchsize should equal the number of input words")
        if contextwordindex.shape[0] != batch_size:
            raise Exception("The batchsize should equal the zeroeth dimension of context words")
        
        # Create negative samples
        loss = 0;
        negsamples = self.generate_negative_samples(negsamplesize, contextwordindex)

        # EJC is the gradients for the output weights w.r.t. the context words.
        # EJN is the gradients for the output weights w.r.t. the negative sample words.
        # Since the input word for each batch can be different, the weights have to updated for the gradient
        # for each batch cumulatively. Since the context and input word indices can vary from batch to batch
        # the gradients are also stored in the batch dimension.
        EJC = np.zeros((batch_size, contextwordindex.shape[0], self.N), dtype=float)
        # EH is the gradients for the hidden weights w.r.t. the input words.
        EJN = np.zeros((batch_size, negsamplesize, self.N), dtype=float)
        EH = np.zeros((batch_size, self.N), dtype=float)

        # Per batch, calculate the output weight gradients for the context words.
        for batch in range(batch_size):
            for i in range(contextwordindex.shape[1]): # For the number of context words in that batch
                # For context words, the target is 1. The "val" is common for both EJ and EH vectors
                
                val = self.sigma(np.dot(self.hid_weights[inputwordindex[batch]], 
                                   self.out_weights[contextwordindex[batch,i]])) - 1
                #val = inp - 1.0
                #if inp > 0:
                #    loss += -log(inp) # Loss computation
                #else:
                #    loss += 6 # very high loss
                loss += abs(val)         
                EJC[batch,i] = val * self.hid_weights[inputwordindex[batch]]
                EH[batch] += val * self.out_weights[contextwordindex[batch,i]]
                
        # Per batch, calculate the output weight gradients for the negative sample words.
        for batch in range(batch_size):
            for i in range(negsamplesize):
                # For negative sampling words, the target is 0. The "val" is common for both EJ and EH vectors
                val = self.sigma(np.matmul(self.hid_weights[inputwordindex[batch]], 
                                  self.out_weights[negsamples[i]]))
                loss += abs(val)/batch_size
                EJN[batch, i] = val * self.hid_weights[inputwordindex[batch]]
                EH[batch] += val * self.out_weights[negsamples[i]]
                
        for batch in range(batch_size):
            # Updated W' weights for context words
            for i in range(contextwordindex.shape[1]):
                self.out_weights[contextwordindex[batch, i]] -= self.Eta * EJC[batch, i]
          
        for batch in range(batch_size):
            # Update W' weights for negative samples
            for i in range(negsamplesize):
                self.out_weights[negsamples[i]] -= self.Eta * EJN[batch, i]
           
        for batch in range(batch_size):
            # Update W weights for each input word in the batch. 
            # This needs to be divided by the number of context words per batch + number of negative smaples
            sum = contextwordindex.shape[1] + negsamplesize
            self.hid_weights[inputwordindex[batch]] -= (self.Eta * EH[batch])/sum
        
        return loss
                           
    def save_weights(self):
        np.save(self.embedsfile, self.hid_weights)
        np.save(self.contextsfile, self.out_weights)
                
