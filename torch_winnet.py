import torch
import torch.nn as nn

rock_tensor = torch.FloatTensor([1, 0, 0])
paper_tensor = torch.FloatTensor([0, 1, 0])
scissors_tensor = torch.FloatTensor([0, 0, 1])


def decoding_rps(rps):
    return ['rock', 'paper', 'scissors'][rps.argmax()]


train_x = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
train_y = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

model = nn.Sequential(
    nn.Linear(3, 10),
    nn.Sigmoid(),
    nn.Linear(10, 3),
    nn.Sigmoid(),
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    y_hat = model(train_x)
    loss = torch.nn.functional.mse_loss(y_hat, train_y)
    loss.backward()
    optimizer.step()

print('rock: ', decoding_rps(model(rock_tensor)))
print('paper: ', decoding_rps(model(paper_tensor)))
print('scissors: ', decoding_rps(model(scissors_tensor)))
