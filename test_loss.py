import torch
from bi_tempered_loss import *

device = "cpu"

activations = torch.FloatTensor([[-0.5,  0.1,  2.0]]).to(device)
labels = torch.FloatTensor([[0.2, 0.5, 0.3]]).to(device)

# The standard logistic loss is obtained when t1 = t2 = 1.0
loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=1.0, t2=1.0)
print("Loss, t1=1.0, t2=1.0: ", loss)

loss = bi_tempered_logistic_loss(activations=activations, labels=labels, t1=0.7, t2=1.3)
print("Loss, t1=0.7, t2=1.3: ", loss)
