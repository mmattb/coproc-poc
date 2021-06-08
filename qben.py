import matplotlib.pyplot as plt

import torch
import torch.autograd
from torch.optim import AdamW, SGD, Adam
import torch.nn

import blah
import stim
import stim_model
import cpn_model

torch.autograd.set_detect_anomaly(True)

loss = torch.nn.MSELoss()

steps = 200
target = torch.zeros((256, steps))
for bidx in range(256):
    for i in range(-100, 100):
        offset = bidx / 256.0
        t = 6 * (i / 100.0) + offset
        target[bidx, i + 100] = torch.sin(torch.tensor(t))

cpn = cpn_model.CPNModel(2, 1, num_neurons=7, activation_func=torch.nn.Tanh)
ben = stim_model.StimModel(1 + 1, 1, num_neurons=7, activation_func=torch.nn.PReLU)



class Thing(object):
    def __init__(self, init=0.0, decay=0.7):
        assert isinstance(init, float)
        self.x = init
        self.decay = decay

    def observe(self):
        return self.x

    def step(self, input):
        assert isinstance(input, float)
        self.x += input
        self.x = self.decay * self.x


opt = AdamW(ben.parameters(), lr=5e-3)
for eidx in range(100):
    ben.reset()
    opt.zero_grad()

cpni = []
opt = AdamW(list(cpn.parameters()) + list(ben.parameters()), lr=6e-3)
for eidx in range(600):
    cpn.reset()
    ben.reset()
    opt.zero_grad()
    thing = [Thing(init=x.item()) for x in target[:,0]]

    cur = target[:,0].reshape(256,1)
    pred = torch.zeros((256, steps))
    s = stim.StimulusGaussianExp(1, 1, batch_size=256, pad_right_neurons=1)

    obs = [t.observe() for t in thing]
    new_obs = torch.tensor(obs).reshape(256,1)

    for tidx in range(steps):
        cpn_in = torch.cat((new_obs, target[:,tidx].reshape(256,1)), axis=1)
        new_stim = cpn(cpn_in)

        s.add(new_stim)
        n = s.get_next()
        #hist[:, tidx] = n[:,:1].squeeze()

        obs = []
        for bidx in range(256):
            thing[bidx].step(n[bidx, 0].item())
            obs.append(thing[bidx].observe())

        new_obs = torch.tensor(obs).reshape(256,1)
        ben_in = torch.cat((new_obs, new_stim), axis=1)
        cur_pred = ben(ben_in)
        pred[:, tidx] = cur_pred.squeeze()

    rl = loss(pred, target)
    print(eidx, rl.item())
    print(torch.max(new_stim), torch.min(new_stim))
    rl.backward()
    opt.step()
    cpni.append(cpn.I.clone())

plt.plot(pred[0,:].detach().numpy())
plt.plot(target[0,:].detach().numpy())
plt.show()

import h5py
f = h5py.File('blah.hdf5', 'w')
cpnic = torch.cat(cpni)
f['blah'] = cpnic.detach().numpy()
f.close()
