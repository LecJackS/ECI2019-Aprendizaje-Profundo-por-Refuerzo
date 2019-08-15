from collections import deque
import torch
import torch.nn.functional as F
import model
from model import SimpleActorCriticLineal
from env import create_train_env
# Actor Critic NN model
AC_NN_MODEL = SimpleActorCriticLineal
ACTOR_HIDDEN_SIZE=256
CRITIC_HIDDEN_SIZE=256

# Test agent
# No learning, no gradients, no dropout (if any)
def local_test(index, opt, global_model, model_type=None):
    torch.manual_seed(42 + index)
    env, num_states, num_actions = create_train_env(opt.layout, index+1, index=index)
    if model_type:
        AC_NN_MODEL = getattr(model, model_type)()
    else:
        AC_NN_MODEL = SimpleActorCriticLineal

    local_model = AC_NN_MODEL(num_states, num_actions)
    # Test model we are going to test (turn off dropout, no backward pass)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            # Copy global model to local model
            local_model.load_state_dict(global_model.state_dict(), strict=False)
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, ACTOR_HIDDEN_SIZE), dtype=torch.float)
                c_0 = torch.zeros((1, CRITIC_HIDDEN_SIZE), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        value = value.clamp(-1.,1.)
        policy = F.softmax(logits, dim=0)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        state = torch.from_numpy(state)
        actions.append(action)

        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = torch.from_numpy(env.reset())
        