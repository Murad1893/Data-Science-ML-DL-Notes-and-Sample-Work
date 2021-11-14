### Common mistakes in PyTorch:
***
- Try to overfit a single batch, to check if loss is decreasing
- Toggle train and eval mode when checking the accuracy (as Dropout behaves differently for both)
- Remember to put optimizer.zero_grad() because we want to run optimizer on the current batch instead of running it on accumulated gradients from all the batches
- No need to do nn.Softmax() when using nn.CrossEntropyLoss() as it automatically does it
- When using batch_norm after a layer, such as conv or linear, we can set the bias to false, as it is already included in BatchNorm2d()
- View and permute are different as shown below, where the first is just a reshape while the second one is the transpose

```python
x = torch.tensor([[1,2,3],[4,5,6]])
print(x.shape)
print(x.view(3,2))
print(x.permute(1,0))
```

```
torch.Size([2, 3])
tensor([[1, 2],
        [3, 4],
        [5, 6]])
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```
- When doing data augmentation, we careful that augmentation does not change the data itself such as vertical flipping "2" would make a nonsense character which is not a digit and hence training the model on that could hurt performance
- Always remeber to shuffle the data. However, in case of a time-series data where the sequence is important, avoid it 🙂
- Always normalize the data with mean 0 and std 1. Less important when using batch norm but still significant in helping the model learn and generalize better
- Remebers to use gradient clipping in case of exploding or vanishing gradients when using RNNs such as LSTMs or GRUs etc. which can be hard to identify or debug..

```python
# forward
scores = model(data)
loss = criterion(scores, targets)

# backward
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)

# adam step
optimizer.step()
```
