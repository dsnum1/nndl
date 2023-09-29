

import torch



class Neuron:
    def __init__(self):
        self.w = torch.tensor([2.5, -0.2, 1.0])
        self.b = torch.tensor(-0.5)

    def __call__(self,x):
        u = torch.inner(self.w, x)+self.b
        y = 0.8/(1+torch.exp(-1.2*u))
        return u,y
    


if __name__ == '__main__':
    n = Neuron()
    u,y = n(torch.tensor([0.1,1.0,1.1]))
    print(y)