import torch.nn as nn

class CountFingersModel(nn.Module):
    def __init__(self):
        super(CountFingersModel, self).__init__()

        # Định nghĩa một mạng neuron đơn giản
        self.linear = nn.Linear(63,31)
        self.reLU = nn.ReLU()
        self.linear2 = nn.Linear(31, 15)
        self.reLU2 = nn.ReLU()
        self.out = nn.Linear(15, 6)


    def forward(self, x):
        x = self.linear(x)
        x = self.reLU(x)
        x = self.linear2(x)
        x = self.reLU2(x)
        x = self.out(x)

        return x