import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import CustomMLP
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# train
loadfile= pd.read_csv('data370.csv')
data_frames = pd.DataFrame(loadfile)

x_frames = data_frames.iloc[:,0:43]
target_frames = data_frames.iloc[:,43:44]

x_array = np.array(x_frames.values)
y_array = np.array(target_frames.values)

X = torch.FloatTensor(x_array).to(device)
Y = torch.FloatTensor(y_array).to(device)


# Split train and test dataset -> train:test = 8:2
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 1/5, random_state=0)



loadfile_new= pd.read_csv('./data.csv')
data_frames_new = pd.DataFrame(loadfile_new)

new_var = data_frames_new.iloc[0,0:43]
new_var2 = np.array(new_var.values)
new_var3 = torch.FloatTensor(new_var2).to(device)
print(new_var3)


# MLP model
model = CustomMLP().to(device)

# criterion = torch.nn.BSELoss().to(device)
criterion = torch.nn.MSELoss().to(device)          #MSE
# criterion = torch.nn.CrossEntropyLoss().to(device)

# weight_decay 적용 -> overfitting 방지
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.0000005)#, weight_decay=0.01)  # modified learning rate from 0.1 to 1


# Train and test
trainloss = []
testloss = []
training_epoches = 14501

for epoch in range(training_epoches):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(X_train)
    cost = criterion(hypothesis, Y_train)                   #MSE
    # cost = torch.sqrt(criterion(hypothesis, Y_train))     #RMSE
    cost.backward()
    optimizer.step()

    # 100의 배수에 해당되는 에포크마다 비용을 출력
    if epoch % 100 == 0:
        print('train:', epoch + 1, cost.item())
    trainloss.append(cost.item())

    # model.eval()
    with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.

        prediction = model(X_test)
        cost_t = criterion(prediction, Y_test)                  #MSE
        # cost_t = torch.sqrt(criterion(prediction, Y_test))  #RMSE

        if epoch % 100 == 0:
            print('test:', epoch + 1, cost_t.item())
        testloss.append(cost_t.item())


# # average loss for training vs. epochs 그래프
x = np.linspace(1, training_epoches, training_epoches)


plt.figure()
plt.plot(x.astype(int), trainloss, label='train_loss')
plt.plot(x.astype(int), testloss, label='test_loss')
plt.title('Average loss vs. epochs (CustomMLP with weight decay)')
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.legend()
plt.show()



#임의의 하중경로 데이터 입력...해당 케이스의 최소 두께 출력
pred_y = model(new_var3)
print("임의의 하중데이터 입력 시 예측된 최소 두께 :", pred_y)




pred_ysss = model(X_train)

Y_train = Variable(torch.Tensor(Y_train.detach().cpu().numpy()))

pred_ysss = Variable(torch.Tensor(pred_ysss.detach().cpu().numpy()))
# Y_train = torch.from_numpy(Y_train.values)



# print('Y_train',Y_train)
# print('pred_ysss',pred_ysss)

# scatter
t = np.arange(0,0.5, 0.05)

plt.figure()

plt.scatter(pred_ysss, Y_train,s=10,marker='o')
plt.plot(t,t,"--",c="red")
# plt.figure(figsize=(10, 6))
plt.xlim(0.29,0.36)
plt.ylim(0.29,0.36)
plt.title('Trained data vs. Predicted data (MLP)')
plt.xlabel('Data_train')
plt.ylabel('Data_predict')
plt.show()
#
#
# # Scatter
# # plt.figure()
# #
# # plt.scatter(pred_ysss,Y_train )
# # plt.plot([2,2],[4,4],c='r',linestyle='--')
# # plt.title('Scatter graph')
# # plt.xlabel('Data_train')
# # plt.ylabel('Data_predict')
