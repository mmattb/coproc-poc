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

cpn = cpn_model.CPNModel(1, 1, num_neurons=4)
opt = AdamW(cpn.parameters(), lr=0.1)

for eidx in range(2000):
    cpn.reset()
    opt.zero_grad()

    cur = target[:,0].reshape(256,1)
    hist = torch.zeros((256, steps))
    s = stim.StimulusGaussianExp(1, 1, batch_size=256, pad_right_neurons=1)

    for tidx in range(steps):
        new_stim = cpn(target[:, tidx].reshape(256,1))

        s.add(new_stim)
        n = s.get_next()
        hist[:, tidx] = n[:,:1].squeeze()

    rl = loss(hist, target)
    print(eidx, rl.item())
    rl.backward()
    opt.step()

plt.plot(hist[0,:].detach().numpy())
plt.plot(target[0,:].detach().numpy())
plt.show()
