import torch
import torch.nn as nn

#nuestro modelo de red neuronal va a tener 3 capas con capas reLu entre ellas
class RedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RedNeuronal, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # Sin capas de activaction o softmax
        return out