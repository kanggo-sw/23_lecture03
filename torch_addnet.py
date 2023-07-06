import torch

train_x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = torch.FloatTensor([[0], [1], [1], [2]])

model = torch.nn.Sequential(torch.nn.Linear(2, 1))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    y_hat = model(train_x)
    loss = torch.nn.functional.mse_loss(y_hat, train_y)
    loss.backward()
    optimizer.step()

test_input = torch.FloatTensor([[1, 1]])
print(model(test_input))
