import torch.nn.functional as F
import torch.distributions
from model import invlink_uniform
import torch

# Loss function for classification tasks
class ce_with_prior():

    def __init__(self, N_train):
        self.N_train = N_train
        print("# Constructing cross-entropy loss")
    
    def train_loss(self, model, input, target):
        return self.loss(model, input, target)

    def test_loss(self, model, input, target):
        output = model(input)
        log_lik = F.cross_entropy(output, target, reduction = 'sum')
        return log_lik/output.shape[0], output

    def loss(self, model, input, target):

        log_lik, output = self.test_loss(model, input, target)

        length_loss = model.l_dist.log_prob(model.lengthscale.forward())

        req_loss = torch.sum(model.Pw_dist.log_prob(model.fc_h.weight)) 
        req_loss += torch.sum(model.Pw_o_dist.log_prob(model.fc_o.weight))
        
        if model.fc_h.bias is not None:
            req_loss += torch.sum(model.Pb_dist.log_prob(invlink_uniform(model.fc_h.bias)))

        L2_loss = 0
        for param in filter(lambda p: p.requires_grad, model.pipeline.parameters()):
            L2_loss -= torch.sum(param**2)
            
        logprior = -(req_loss + L2_loss + length_loss)/self.N_train
        loss = log_lik + logprior

        return loss, output
        
# Loss function for regression tasks
class L2_with_prior():
        
    def __init__(self, N_train):
        self.N_train = N_train
        print("# Constructing cross-entropy loss")
    
    def train_loss(self, model, input, target):
        return self.loss(model, input, target)

    def test_loss(self, model, input, target):
        output = model(input)
        if model.s is not None:
            s = model.s.forward()
            N_dist = torch.distributions.normal.Normal(0,s)
            log_lik = -1*torch.sum(N_dist.log_prob(output.squeeze() - target.squeeze()))
        else:
            log_lik = torch.sum((output.squeeze() - target.squeeze())**2)
        return log_lik/output.shape[0], output

    def loss(self, model, input, target):
            
        log_lik, output = self.test_loss(model, input, target)

        length_loss = model.l_dist.log_prob(model.lengthscale.forward())
        if model.s is not None:
            meas_noise_loss = model.s_dist.log_prob(model.s.forward())
        else:
            meas_noise_loss = 0

        req_loss = torch.sum(model.Pw_dist.log_prob(model.fc_h.weight)) 
        req_loss += torch.sum(model.Pw_o_dist.log_prob(model.fc_o.weight))
        
        if model.fc_h.bias is not None:
            req_loss += torch.sum(model.Pb_dist.log_prob(invlink_uniform(model.fc_h.bias)))

        L2_loss = 0
        for param in filter(lambda p: p.requires_grad, model.pipeline.parameters()):
            L2_loss -= torch.sum(param**2)
            
        logprior = -(req_loss + L2_loss + length_loss + meas_noise_loss)/self.N_train
        loss = log_lik + logprior

        return loss, output
