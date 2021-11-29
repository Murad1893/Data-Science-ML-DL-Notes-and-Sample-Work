import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, trainCNN=False):
        super(EncoderCNN, self).__init__()
        self.trainCNN = trainCNN
        self.inception = models.inception_v3(pretrained=True,
                                             aux_logits=False)  # aux_logits only used for loss during training

        # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        # we replace the FC layer with our own layer
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        # however, we are not going train so no grads except for the FC
        for name, param in self.inception.named_parameters():
            # print(name, param)
            # the names are like : Mixed_7c.branch3x3dbl_1.bn.bias etc. so comparing by name
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return self.dropout(self.relu(features))  # adding the dropout and relu


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))  # getting the embeddings

        # concatentating embeddings with image features is often used in image captioning tasks
        embeddings = torch.cat((features.unsqueeze(0), embeddings),
                               dim=0)  # concat the features across the 1st dim (time dimension)
        out, _ = self.lstm(embeddings)
        outputs = self.linear(out)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)

        return outputs


if __name__ == '__main__':
    inception = models.inception_v3(pretrained=False)

    # (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # for name, param in inception.named_parameters():
    #     if "fc.weight" in name:
    #         print(name, param.shape)
