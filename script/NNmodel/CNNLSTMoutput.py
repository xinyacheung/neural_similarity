import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=500, num_layers=6, num_classes=3):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False) #True
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.Linear(256,num_classes)
        )

    def forward(self, x): # [bs, seq_len, hidden_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        out = hidden[0, :, :]
        preds = self.fc(out)

        return preds
    


if torch.cuda.is_available():
    logging.warning("Cuda is available!")
else:
    logging.warning("Cuda is not availble! Exit!")

output_size = 1000 
seq_len = 270
model = (torch.load(f'/home/../model.pth')).cuda()
print(model)

return_layers={
    'outfc.0':'relu',
    'outfc.1':'linear256',
    'outfc.2':'linear3'
}


#Loading Test Data:
for i in range(21):
  x=torch.load(f'LSTMData/LSTMTestX_{i}_{output_size}')
  y=torch.load(f'LSTMData/LSTMTestY_{i}_{output_size}')
  if i==0:
    X=x
    Y=y
  else:
    X=torch.cat((X,x),dim=0)
    Y=torch.cat((Y,y),dim=0)
print('X.shape : ',X.shape) 
print('Y.shape : ',Y.shape)

test_X=X.detach()
test_Y=Y.detach()

#Difference sequence length:
test_X=test_X[:,0:seq_len,:] 
col=np.array([-90,-30,-5,0,5,30,90]);
cl={-90: "#5ead93", -30: "#a0a762", 30: "#de9765", 90: "#e1818d",0: "#bc9c58",-5:"#b89e58",5:"#c19b58"}
#Possible combinations of directions and colors (21 points):
comb1 = torch.tensor([[0,0],[-5,0],[5,0],[0,-5],[0,5]])
comb2=torch.tensor([[90,90],[90,30],[90,-30],[90,-90],[30,90],[30,30],[30,-30],[30,-90],[-30,90],[-30,30],[-30,-30],
                [-30,-90],[-90,90],[-90,30],
                [-90,-30],[-90,-90]])
comb=torch.cat((comb1,comb2),dim=0)

combinations=torch.zeros(80*21,1)
for i in range(21):
  combinations[80*i:80*(i+1)]=i
combinations=combinations.squeeze(1)



act=torch.zeros(21,80,1000)  #256 is the size of LSTM activations
CC=torch.zeros(21,40,1000)
CM=torch.zeros(21,40,1000)

for index in range(21):
    testset = TensorDataset(test_X[80*index:80*(index+1),:,:], test_Y[80*index:80*(index+1)],combinations[80*index:80*(index+1)])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False)
    countCol=0
    countMot=0
    with torch.no_grad():
        for batch_idx,(data,targets,c) in enumerate(testloader):
            mid_getter=MidGetter(model,return_layers=return_layers,keep_output=True)
            activations,model_output=mid_getter(data.cuda())
            a=(activations['linear256']) 
            act[index,batch_idx,:]=a.cpu()
            if (batch_idx <20) or (40<=batch_idx<60):
                CM[index,countMot,:]=a    # 1 speed * 2cue * 10 times +  1 speed * 2cue * 10 times
                countMot += 1
            else:
                CC[index,countCol,:]=a
                countCol +=1


#Taking the average:
CMavg=(CM.sum(dim=1))/40
CCavg=(CC.sum(dim=1))/40

# Distance Matrix : Motion Condition
DMmotion = np.zeros((21, 21))
for row in range(21):
    for col in range(21):
        v1 = CMavg[row]  
        v2 = CMavg[col]  
        dist = np.linalg.norm(v1 - v2)
        # dist = distance.cosine(v1,v2) 
        DMmotion[row, col] = dist
cmap = plt.get_cmap('Blues')
matrix = DMmotion[5:, 5:]
torch.save(matrix,f'/home/../model_DMmotion.pt')

# Distance Matrix : Colour Condition
DMcolour = np.zeros((21, 21))
for row in range(21):
    for col in range(21):
        v1 = CCavg[row]
        v2 = CCavg[col] 
        dist = np.linalg.norm(v1 - v2)
        # dist = distance.cosine(v1,v2) 
        DMcolour[row, col] = dist
        print(dist)
cmap = plt.get_cmap('Blues')
matrix = DMcolour[5:, 5:]
torch.save(matrix, f'/home/../model_DMcolor.pt')
