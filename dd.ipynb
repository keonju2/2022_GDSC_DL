import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
# 데이터를 훈련용과 테스트용으로 분리
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 전체의 20%는 검증용
digits = load_digits()
X = digits.data
Y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.int64).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.int64).to(device)



linear1 = torch.nn.Linear(64, 100, bias=True)
linear2 = torch.nn.Linear(100, 100, bias=True)
linear3 = torch.nn.Linear(100, 100, bias=True)
linear4 = torch.nn.Linear(100, 100, bias=True)
linear5 = torch.nn.Linear(100, 10, bias=True)
relu = torch.nn.ReLU()


torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)


model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3,
                           linear4,relu,linear5).to(device)


lossFunc = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

ds = TensorDataset(X_train, Y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True)

trainLosses = []
testLosses = []

for epoch in tqdm(range(100)):
    runningLoss = 0.0
    # 신경망을 train 모드로 설정
    model.train()
    for i, (batchX, batchY) in enumerate(loader):
        optimizer.zero_grad()
        
        yPred = model(batchX)
        loss = lossFunc(yPred, batchY)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    trainLosses.append(runningLoss/i)
    
    # 신경망을 test 모드로 설정
    model.eval()
    yPred = model(X_test)
    testLoss = lossFunc(yPred, Y_test)
    testLosses.append(testLoss.item())

plt.plot(range(100), trainLosses, label = "train loss")
plt.plot(range(100), testLosses, label = "test loss")
plt.legend()
