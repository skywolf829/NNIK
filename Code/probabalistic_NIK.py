from random import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax
from kinematics import random_configuration, switch_config_order
from torch.utils.tensorboard import SummaryWriter
import os
from kinematics_modules import FK

class primary_network(nn.Module):
    '''
    This is the so-called "primary network", which is responsible for
    the GMM for the likelyhood of angles for a single arm given
    input previous arm angles. This model is not directly learning
    anything. Instead, the weights supplied to it come from the
    hypernetwork (defined in probablistic_NIK), which is conditioned
    on the desired end point of the n segment arm.
    
    This model is 3 layers deep with an output size of 3*n_GMM,
    which represents a GMM of n_GMM gaussians. The 3 components are
    the mean, variance, and mixture coefficient (pi_k) for each 
    gaussian.
    
    As a note, the first arm (attached to the base) does not take
    any inputs, but the network still starts with 1 "dummy" input
    to hold place. In the hypernetwork, the input is set to 0
    so the weights are ignored. However, the bias terms of the
    first layer are still relevant.
    '''
    def __init__(self, n_inputs, 
                 actuator_range,
                 n_GMM=50, 
                 nodes_per_layer=256):
        super().__init__()
        
        self.n_GMM = n_GMM
        self.actuator_range = actuator_range
        self.n_inputs = n_inputs
        self.nodes_per_layer = nodes_per_layer
        
        # Used in forward pass for the mixing coefficients of the
        # gaussians in the GMM. More details in forward()
        self.sparsemax = Sparsemax(dim=-1)        
        
        self.init_weights_biases()
        
    def init_weights_biases(self, batch_size=1):
                
        # For the first arm, there must be 1 "dummy" input node
        # and the input is always 0 so the weights are ignored.
        self.weights1 = torch.empty([batch_size, self.nodes_per_layer, max(1, self.n_inputs)],
                                    dtype=torch.float32)
        self.bias1 = torch.empty([batch_size, self.nodes_per_layer, 1],
                                 dtype=torch.float32)
        
        self.weights2 = torch.empty([batch_size, self.nodes_per_layer, self.nodes_per_layer],
                                    dtype=torch.float32)
        self.bias2 = torch.empty([batch_size, self.nodes_per_layer, 1],
                                 dtype=torch.float32)
        
        self.weights3 = torch.empty([batch_size, self.nodes_per_layer, self.nodes_per_layer],
                                    dtype=torch.float32)
        self.bias3 = torch.empty([batch_size, self.nodes_per_layer, 1],
                                 dtype=torch.float32)
        
        self.weights4 = torch.empty([batch_size, 3*self.n_GMM, self.nodes_per_layer],
                                    dtype=torch.float32)
        
    def set_weights(self, weight_array):
        '''
        A utility function for the hypernetwork to easily
        update the weights of this network given the weight_array.
        '''
        spot = 0
    
        # Update the batch size if necessary
        b = weight_array.shape[0]
        if(self.weights1.shape[0] != b):
            self.init_weights_biases(b)
                
        # weights1
        weight_size = self.weights1.shape[1]*self.weights1.shape[2]
        self.weights1 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.weights1.shape)
        spot += weight_size
        # bias1
        weight_size = self.bias1.shape[1]*self.bias1.shape[2]
        self.bias1 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.bias1.shape)
        spot += weight_size
        
        # weights2
        weight_size = self.weights2.shape[1]*self.weights2.shape[2]
        self.weights2 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.weights2.shape)
        spot += weight_size
        
        # bias2
        weight_size = self.bias2.shape[1]*self.bias2.shape[2]
        self.bias2 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.bias2.shape)
        spot += weight_size
        
        # weights3
        weight_size = self.weights3.shape[1]*self.weights3.shape[2]
        self.weights3 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.weights3.shape)
        spot += weight_size
        
        # bias3
        weight_size = self.bias3.shape[1]*self.bias3.shape[2]
        self.bias3 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.bias3.shape)
        spot += weight_size
        
        # weights4
        weight_size = self.weights4.shape[1]*self.weights4.shape[2]
        self.weights4 = weight_array[
                    :,spot:spot+weight_size
                ].reshape(self.weights4.shape)
        spot += weight_size
             
    def forward(self, x):
        '''
        Computs the output GMM for the input x.
        Expects x to be [batch, n_inputs]
        '''
        
        x = torch.bmm(self.weights1, x.unsqueeze(-1))
        x += self.bias1
        x = F.relu(x)
               
        x = torch.bmm(self.weights2, x)
        x += self.bias2
        x = F.relu(x)
        
        x = torch.bmm(self.weights3, x)
        x += self.bias3
        x = F.relu(x)
        
        x = torch.bmm(self.weights4, x)[...,0]
        
        # Restrict the means to the range for 
        # the actuator this represents
        x[:,0::3] = (torch.tanh(x[:,0::3])+1.0)/2.0 * self.actuator_range
        
        # Variance cannot be negative, so use torch.abs to fix 
        # any negative variance scores
        x[:,1::3] = torch.abs(x[:,1::3].clone())
        
        # Apply sparsemax to the mixing coefficients.
        # Install with 'pip install sparsemax'
        # It is a more selective attention based softmax, so 
        # results sum to 1, but the focus is only on a few of the gaussians.
        x[:,2::3] = self.sparsemax(x[:,2::3])
        
        return x

class probabalistic_NIK(nn.Module):  
    '''
    The hypernetwork for the probabalistic NIK paper.
    The model is 4 layers deep, with N "projection layers",
    which are simply separated linear layers that are responsible
    for mapping the hypernetwork's output to each primary
    networks weights. Since the primary networks may have
    different sizes (just the first one), the projection
    layers may have different sizes too. 
    '''
    def __init__(self, n_segments=3, 
                 config_per_segment=2, 
                 n_GMM=50, 
                 nodes_per_layer = 1024,
                 primary_network_nodes_per_layer=256):
        super().__init__()        
        
        self.n_segments = n_segments
        self.config_per_segment = config_per_segment
        self.n_GMM = n_GMM
        self.nodes_per_layer = nodes_per_layer
        self.primary_network_nodes_per_layer = primary_network_nodes_per_layer
        
        # The main component that maps a desired tip position
        # to the deep latent vector that is used as input to
        # the projection layers to create the weights for the 
        # primary networks.
        print("Creating main component")
        self.main_component = nn.Sequential(
            nn.Linear(3, self.nodes_per_layer, dtype=torch.float32),
            nn.ReLU(),
            nn.BatchNorm1d(self.nodes_per_layer, dtype=torch.float32),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer, dtype=torch.float32),
            nn.ReLU(),
            nn.BatchNorm1d(self.nodes_per_layer, dtype=torch.float32),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer, dtype=torch.float32),
            nn.ReLU(),
            nn.BatchNorm1d(self.nodes_per_layer, dtype=torch.float32),
            nn.Linear(self.nodes_per_layer, self.nodes_per_layer, dtype=torch.float32),
            nn.ReLU(),
            nn.BatchNorm1d(self.nodes_per_layer, dtype=torch.float32)
        )
        print(self.main_component)
        
        # Separate projection layers for the primary networks
        self.projection_layers = nn.ModuleList()
        self.primary_networks = nn.ModuleList()
        
        # Create one projection layer and one primary network
        # for each configuration output per arm.
        # For a rigid arm, there is one configuration item per arm
        # which is just the rotation. For a soft robotic arm,
        # we have two each. It is assumed the output is in
        # theta, phi order, since phi will have more reliance
        # on theta per segment.
        print("Creating each projection layer and primary network")
        for i in range(self.n_segments*self.config_per_segment):
            self.projection_layers.append(
                nn.Linear(self.nodes_per_layer, # the output from the main component 
                          max(1,i)*self.primary_network_nodes_per_layer + \
                              self.primary_network_nodes_per_layer + # input to first layer
                          self.primary_network_nodes_per_layer*self.primary_network_nodes_per_layer + \
                              self.primary_network_nodes_per_layer + #first to second layer
                          self.primary_network_nodes_per_layer*self.primary_network_nodes_per_layer + \
                              self.primary_network_nodes_per_layer + #second to third layer
                          self.primary_network_nodes_per_layer*(3*self.n_GMM) #third to output
                          , dtype=torch.float32)
            )
            self.primary_networks.append(
                primary_network(i, 
                                2*torch.pi if i%2 == 0 else torch.pi,
                                n_GMM = self.n_GMM)
            )

    def parameters(self):
        return self.main_component.parameters()

    def sample_gmm(self, gmm):
        # Given that gmm[:,2::3] represent the mixture weights,
        # they are used as the probabilities to draw from gmm_i.
        # torch.multinomial takes in an array of weights, and takes 
        # n samples from those weights, and returns the sampled
        # index. Below, will sample one gmm index for each item
        # in the batch given.
        sampled_gmm = torch.multinomial(gmm[:,2::3], 1)[:,0]*3
        
        # Next, we create a randomly drawn sample from the
        # gaussian for each index sampled using each
        # ones mean and variance
        mu = gmm[range(gmm.shape[0]),sampled_gmm+0]
        sigma = gmm[range(gmm.shape[0]),sampled_gmm+1]**0.5
        sampled = torch.normal(mu, sigma)
        return sampled        
    
    def forward(self, x, config=None):
        '''
        Samples the gmms from the primary networks
        given a desired tip position x to create a 
        configuration, which is returned. 
        
        One sample per batch element in x is generated,
        so if multiple samples for a single tip
        position are required, use torch.repeat
        to repeat the desired position in the first
        dimension of x, to make it of shape [n, 1].
        
        If config is None, the method assumes you want
        to sample a config for the given tip position x.
        The GMMs will be sampled and a sampled configuration
        will be returned for the batch.
        Otherwise, a correctly sized configuration tensor
        of shape [n, self.n_segments] is assumed to be training
        examples, and a log likelyhood of the configuration
        is returned.
        '''
        # Get the deep latent vector
        pre_projection = self.main_component(x)
        
        # this is the sampled position from each GMM, built from
        # the base upward, because each further component is
        # reliant on the previous configuration choices
        if(config is None):
            configuration = []
        else:
            log_likelyhood = 0
            
        # Sample configurations from the bottom upward
        for i in range(len(self.projection_layers)):
            
            # Get the weights for the primary network and update them
            weights = self.projection_layers[i](pre_projection)
            self.primary_networks[i].set_weights(weights)
            
            # Create the input for the primary network, which should
            # be the configuration of the previous joints
            # A special case is the first arm, which takes no inputs,
            # so we set the dummy input to 0
            
            primary_network_input = torch.zeros([x.shape[0],1], device=weights.device)
            if(i > 0 and config is None):
                primary_network_input[:] = torch.tensor(configuration, device=weights.device)
            elif(i > 0):
                primary_network_input = config[:,0:i].clone().to(weights.device)
                
            # Get the GMM output for the network
            # gmm_i[:,0::3] is the mean, 
            # gmm_i[:,1::3] is the variance, and
            # gmm_i[:,2::3] is the mixing coefficient
            gmm_i = self.primary_networks[i](primary_network_input)
            
            if(config is None):
                # Sample from the final GMM
                sampled_gmm = self.sample_gmm(gmm_i)
            else:
                # The teacher forcing sample config for this 
                # segment 
                sample = config[:,i:i+1]
                
                '''
                # calculate the log likelyhood of this sample given
                # the gmm_i the networks have created
                gmms = torch.distributions.normal.Normal(
                    gmm_i[:,0::3], 
                    gmm_i[:,1::3]**0.5)
                ll_i = gmms.log_prob(sample).sum(dim=1)         
                
                # add that to the total log likelyhood
                log_likelyhood += ll_i
                '''
                
                ll_i = (-1/2) * (torch.log(2*torch.pi*gmm_i[:,1::3]) + (1/gmm_i[:,1::3])*((sample-gmm_i[:,0::3])**2))
                log_likelyhood += ll_i.sum(dim=1)
        
        
        if(config is None):
            return sampled_gmm
        else:
            return log_likelyhood
    
def train(model, iterations=10000, batch_size=100, lr=0.05, device="cuda:0"):
    '''
    Trains the given model using random sampling of the 
    configuration space.
    '''
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(os.path.join('tensorboard'))
    
    fk = FK(3)

    print("Beginning training")
    for iter in range(iterations):
        model.zero_grad()
        
        # Sample random configurations
        random_configs = random_configuration(batch_size, 3, device)
        # Find the fk result for these configurations
        fk_out = fk(random_configs)        
        #Fix the ordering to theta-phi for training        
        random_configs = switch_config_order(random_configs)
        
        # Get the log-likelyhood of the configurations given the tip position
        # of the current model, which we want to minimize
        ll = -model(fk_out, config=random_configs).sum()
        
        print(f"Iteration {iter+1}/{iterations} - ll:{ll.item() : 0.04f}")
        writer.add_scalar('log_likelyhood', ll.item(), iter)

        # Update the model
        ll.backward()
        optim.step()
    
    return model

if __name__ == "__main__":
    np.random.seed(0)       
    torch.random.manual_seed(0)    
    device = "cuda"
    
    model = probabalistic_NIK()
    model = train(model)
    