# Se definen optimizarores con parámetros compartidos con los globales, para aplicar gradientes
# desde procesos locales, a parámetros globales.
import torch

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                #state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class GlobalRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr):
        super(GlobalRMSprop, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = p.data.new().resize_(1).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

                state['step'].share_memory_()
                state['square_avg'].share_memory_()