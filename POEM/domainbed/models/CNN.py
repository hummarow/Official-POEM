import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_conv=2):
        """
        Convolution block
        :param num_conv: Convolution Block 속 convolution layer 개수. Default: 2.
        :type num_conv: int
        """
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"))
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_conv - 1):
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding="same")
            )
            layers.append(nn.ReLU(inplace=True))
        #         layers.append(nn.MaxPool2d(3,2, padding=(out_channels)))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=1, out_features=3):
        super().__init__()
        # backbone network
        conv1 = ConvBlock(1, 4, num_conv=1)
        conv2 = ConvBlock(4, 16, num_conv=1)
        # conv3 = ConvBlock(16, 16, num_conv=1)

        self.flat = nn.Flatten()
        dense1 = nn.Sequential(
            nn.Linear(1536, 32),
            nn.ReLU(inplace=True),
        )
        # dense2 = nn.Sequential(nn.Linear(1536, 1536),
        #                             nn.ReLU(inplace=True),
        #                             )
        dense2 = nn.Linear(32, out_features)

        self.featurizer = nn.Sequential(
            conv1,
            conv2,
            # conv3,
            self.flat,
        )
        self.featurizer.n_outputs = 1536
        self.dense = nn.Sequential(
            dense1,
            dense2,
            #    dense3,
        )

    def forward(self, x):
        f = self.featurizer(x)
        v = self.dense(f)
        return v

    def _init_params(self):
        print(self.modules())
        exit()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_style(self, x):
        mu = x.mean(dim=[2, 3])
        sig = x.std(dim=[2, 3])
        return torch.cat([mu, sig], 1)


class RNN(nn.Module):
    def __init__(self, in_channels=1, out_features=3):
        super().__init__()

        class RNNWrapper(nn.Module):
            """
            Extract and return only the output from a tuple, which is returned by an RNN.
            This class should be designed to be used within an nn.Sequential.
            """

            def __init__(self, rnn):
                super().__init__()
                self.rnn = rnn

            def forward(self, x):
                return self.rnn(x)[0][:, -1, :]

        class Squeeze(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.squeeze(x, 1)

        self.flat = nn.Flatten()
        self.featurizer = nn.Sequential(
            Squeeze(),
            RNNWrapper(
                nn.RNN(input_size=63, hidden_size=63, num_layers=2, batch_first=True)
            ),  # (N, L, H_in): (112, 400, 63)
            self.flat,
        )
        dense1 = nn.Sequential(
            nn.Linear(63, 16),
            nn.ReLU(inplace=True),
        )
        dense2 = nn.Linear(16, out_features)

        self.featurizer.n_outputs = 63

        self.dense = nn.Sequential(
            dense1,
            dense2,
            #    dense3,
        )

    def forward(self, x):
        f = self.featurizer(x)
        v = self.dense(f)
        return v


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)
