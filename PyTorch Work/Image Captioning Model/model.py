import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, trainCNN=False):
        super(EncoderCNN, self).__init__()
        self.trainCNN = trainCNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False) # aux_logits only used for loss during training

        # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        # we replace the FC layer with our own layer
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self):
        pass


if __name__ == '__main__':
    inception = models.inception_v3(pretrained=False)

    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    print(inception)