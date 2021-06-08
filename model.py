import torch.nn as nn
import torch.nn.functional as F
import torch


class CustomMLP(nn.Module):
        def __init__(self):
            super(CustomMLP, self).__init__()
            self.layer1 = nn.Linear(43, 50)
            self.layer2 = nn.Linear(50, 50)
            self.layer3 = nn.Linear(50, 50)
            self.layer4 = nn.Linear(50, 1)

        def forward(self, x):
            F = nn.functional
            x = F.relu(self.layer1(x))
            # out = self.dropout(out)
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            # out = self.dropout(out)
            x = self.layer4(x)

            return x

    # def __init__(self):
    #     super(CustomMLP, self).__init__()
    #     self.fc1 = nn.Linear(32 * 32, 50)
    #     self.fc2 = nn.Linear(50, 140)
    #     self.fc3 = nn.Linear(140, 10)
    #     self.dropout = nn.Dropout(0.2)  # avoid overfitting
    #
    # def forward(self, img):
    #     F = nn.functional
    #     out = img.view(-1, 32 * 32)
    #     out = F.relu(self.fc1(out))     # 32*32*50 = 51200
    #     out = self.dropout(out)
    #     out = F.relu(self.fc2(out))     # 50*140 = 7000
    #     out = self.dropout(out)
    #     output = self.fc3(out)          # 140*10 = 1400
    #                                     # total = 51200 + 7000 + 1400 + 50 + 140 + 10 = 59800
    #     return output
    #

# class CustomMLP(nn.Module):
#
#     def __init__(self):
#         super(CustomMLP, self).__init__()
#         self.layer1 = nn.Linear(43, 20)
#         self.layer2 = nn.Linear(20, 20)
#         self.layer3 = nn.Linear(20, 1)
#
#     def forward(self, x_train):
#         F = nn.functional
#         out = x_train.view(-1, 43)
#         out = F.relu(self.layer1(out))
#         # out = self.dropout(out)
#         out = F.relu(self.layer2(out))
#         # out = self.dropout(out)
#         output = self.layer3(out)
#
#         return output