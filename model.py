# NN model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# PyTorch input tensor : [batch_size, channel, height, width]

class SimpleActorCriticLineal(nn.Module):
    """Same as Mnih2016ActorCritic but with dropout layers for each layer,
       and feedforward linear instead of LSTM"""
    def __init__(self, num_inputs, num_actions):
        super(SimpleActorCriticLineal, self).__init__()
        self.p_drop_linear = 0.4
        # Model Representation
        #self.lstm = nn.LSTMCell(num_inputs, 256)
        self.linear_1 = nn.Linear(num_inputs, 64)
        self.linear_2_drop = nn.Dropout(p=self.p_drop_linear)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3_drop = nn.Dropout(p=self.p_drop_linear)
        self.linear_3 = nn.Linear(64, 256)
        #self.lstm = nn.LSTMCell(64, 256)
        # Outputs
        self.actor_drop    = nn.Dropout(p=self.p_drop_linear)
        self.actor_linear  = nn.Linear(256, num_actions)
        self.critic_drop   = nn.Dropout(p=self.p_drop_linear)
        self.critic_linear = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            #elif isinstance(module, nn.LSTMCell):
            #    nn.init.constant_(module.bias_ih, 0)
            #    nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        #hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        # to flatten all filters: x.view(x.size(0), -1)
        #x = F.relu(self.linear(x.view(x.size(0), -1)))
        x = F.relu(self.linear_1(x))
        x = self.linear_2_drop(F.relu(self.linear_2(x)))
        x = self.linear_3_drop(F.relu(self.linear_3(x)))
        #hx, cx = self.lstm(x.view(-1, x.size(0)), (hx, cx))
        # Softmax is applied later, same with log-softmax
        actor  = self.actor_drop(self.actor_linear(x))
        critic = self.critic_drop(self.critic_linear(x))
        # Uncomment to print forward values
        #print("Actor: " , actor)
        #print("Critic: ", critic)
        return actor, critic, hx, cx

class SimpleActorCriticLSTM(nn.Module):
    """Same as Mnih2016ActorCritic but with dropout layers for each layer,
       and feedforward linear instead of LSTM"""
    def __init__(self, num_inputs, num_actions):
        super(SimpleActorCriticLSTM, self).__init__()
        self.p_drop_linear = 0.4
        # Model Representation
        #self.lstm = nn.LSTMCell(num_inputs, 256)
        self.linear_1 = nn.Linear(num_inputs, 64)
        self.linear_2_drop = nn.Dropout(p=self.p_drop_linear)
        self.linear_2 = nn.Linear(64, 64)
        #self.linear_3_drop = nn.Dropout(p=self.p_drop_linear)
        #self.linear_3 = nn.Linear(64, 256)
        self.lstm = nn.LSTMCell(64, 256)
        # Outputs
        self.actor_drop    = nn.Dropout(p=self.p_drop_linear)
        self.actor_linear  = nn.Linear(256, num_actions)
        self.critic_drop   = nn.Dropout(p=self.p_drop_linear)
        self.critic_linear = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        #hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        # to flatten all filters: x.view(x.size(0), -1)
        #x = F.relu(self.linear(x.view(x.size(0), -1)))
        x = F.relu(self.linear_1(x))
        x = self.linear_2_drop(F.relu(self.linear_2(x)))
        #x = self.linear_3_drop(F.relu(self.linear_3(x)))
        hx, cx = self.lstm(x.view(-1, x.size(0)), (hx, cx))
        # Softmax is applied later, same with log-softmax
        actor  = self.actor_drop(self.actor_linear(hx))
        critic = self.critic_drop(self.critic_linear(cx))
        # Uncomment to print forward values
        #print("Actor: " , actor)
        #print("Critic: ", critic)
        return actor, critic, hx, cx

class RedutionActorCriticWithDropout(nn.Module):
    """Same as Mnih2016ActorCritic but with dropout layers for each layer,
       and feedforward linear instead of LSTM"""
    def __init__(self, num_inputs, num_actions):
        super(ReductionActorCriticWithDropout, self).__init__()
        self.p_drop_linear = 0.4
        # Model Representation
        #self.lstm = nn.LSTMCell(num_inputs, 256)
        self.linear_1 = nn.Linear(num_inputs, 64)
        self.linear_2_drop = nn.Dropout(p=self.p_drop_linear)
        self.linear_2 = nn.Linear(64, 32)
        #self.linear_3_drop = nn.Dropout(p=self.p_drop_linear)
        #self.linear_3 = nn.Linear(32, 256)
        self.lstm = nn.LSTMCell(32, 256)
        # Outputs
        self.actor_drop    = nn.Dropout(p=self.p_drop_linear)
        self.actor_linear  = nn.Linear(256, num_actions)
        self.critic_drop   = nn.Dropout(p=self.p_drop_linear)
        self.critic_linear = nn.Linear(256, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        #hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        # to flatten all filters: x.view(x.size(0), -1)
        #x = F.relu(self.linear(x.view(x.size(0), -1)))
        x = F.relu(self.linear_1(x))
        x = self.linear_2_drop(F.relu(self.linear_2(x)))
        #x = self.linear_3_drop(F.relu(self.linear_3(x)))
        hx, cx = self.lstm(x.view(-1, x.size(0)), (hx, cx))
        # Softmax is applied later, same with log-softmax
        actor  = self.actor_drop(self.actor_linear(hx))
        critic = self.critic_drop(self.critic_linear(cx))
        # Uncomment to print forward values
        #print("Actor: " , actor)
        #print("Critic: ", critic)
        return actor, critic, hx, cx