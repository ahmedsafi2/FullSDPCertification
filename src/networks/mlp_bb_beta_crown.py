from torch import nn

def mnist_6_100():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100, 10),
        # nn.ReLU(),
        # nn.Linear(10,10, bias=False)
    )
    return model