import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD
from PIL import Image
import io
from copy import deepcopy
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


parent_save_dir = '/home/user/../data'
#Loading the fixpoint image
imageFP = Image.open(parent_save_dir+'fixpoint.jpg').convert('RGB')
xFP=TF.to_tensor(imageFP)
print(xFP.shape)

#Loading the images of the cues
imageCR = Image.open(parent_save_dir+'cross.jpg').convert('RGB')
xCR=TF.to_tensor(imageCR)
imageQ = Image.open(parent_save_dir+'quatrefoil.jpg').convert('RGB')
xQ=TF.to_tensor(imageQ)
imageCI = Image.open(parent_save_dir+'circle.jpg').convert('RGB')
xCI=TF.to_tensor(imageCI)
imageT = Image.open(parent_save_dir+'triangle.jpg').convert('RGB')
xT=TF.to_tensor(imageT)
print(xCR.shape)
print(xQ.shape)
print(xCI.shape)
print(xT.shape)


#Possible directions:
dir=np.array([-90,-30,-5,0,5,30,90]);
#Possible colors:
col=np.array([-90,-30,-5,0,5,30,90]);
cl={-90: "#5ead93", -30: "#a0a762", 30: "#de9765", 90: "#e1818d",0: "#bc9c58",-5:"#b89e58",5:"#c19b58"}
#Possible combinations of directions and colors (21 points):
comb1 = np.array([[0,0],[-5,0],[5,0],[0,-5],[0,5]])
comb2=np.array([[90,90],[90,30],[90,-30],[90,-90],[30,90],[30,30],[30,-30],[30,-90],[-30,90],[-30,30],[-30,-30],
                [-30,-90],[-90,90],[-90,30],
                [-90,-30],[-90,-90]])
comb=np.append(comb1,comb2,axis=0)
print(len(comb))
Npoints = 400

def gendots(speed_choice,dir_choice):
  #Generate Points:
  r=np.sqrt(np.random.rand(Npoints,1))
  t=2*np.pi*np.random.rand(Npoints,1)
  cs=np.concatenate([np.cos(t),np.sin(t)],axis=1) 
  M=np.concatenate([r,r],axis=1)*cs 
  xprime=M[:,0]
  yprime=M[:,1]
  if speed_choice=='slow':
    speed=1.67/60
  else:
    speed=10/60
  #Movement per frame:
  Dxy=np.tan(speed*np.pi/180)*20
  angle=np.pi*(dir_choice)/180
  Dx=Dxy*np.cos(angle)
  Dy=Dxy*np.sin(angle)


  xcoord=np.zeros((Nimgs,Npoints))
  ycoord=np.zeros((Nimgs,Npoints))
  for img in range(Nimgs):
    newx=xprime+Dx
    newy=yprime+Dy
    outdots=np.where(newx**2+newy**2>1) #adjusting for when dots go outside the circle

    r=np.sqrt(np.random.rand(len(outdots[0]),1))
    t=2*np.pi*np.random.rand(len(outdots[0]),1)
    cs=np.concatenate([np.cos(t),np.sin(t)],axis=1)
    M=np.concatenate([r,r],axis=1)*cs
    newx[outdots]=M[:,0]
    newy[outdots]=M[:,1]

    xprime=newx #the new x inside the circle
    yprime=newy #the new y inside the circle
    xcoord[img,:]=xprime.reshape(Npoints)
    ycoord[img,:]=yprime.reshape(Npoints)

  return xcoord,ycoord


ms = (25 / 3) * (1.28 / 10)

def genimgs(speed_choice, dir_choice, col_choice):
  col_code = cl[col_choice]
  xcoord, ycoord = gendots(speed_choice, dir_choice)
  x = torch.zeros(Nimgs, 3, 128, 128)
  for img in range(Nimgs):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1.28, 1.28)
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    xprime = xcoord[img, :]
    yprime = ycoord[img, :]
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.plot(xprime, yprime, '.', color=col_code, ms=ms)
    ax.plot(0, 0, '.', color='w', ms=ms)
    ax.plot(-1.5, 0, '.', color="#9e9c9f", ms=ms)
    ax.plot(1.5, 0, '.', color="#9e9c9f", ms=ms)
    ax.set_facecolor('k')
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf).convert("RGB"))
    buf.close()
    x[img, :, :, :] = TF.to_tensor(pil_img)
    plt.close('all')

  return x             

# 0->L & 1->R
# Motion: L->up & R->down
# Color: L->green & R->red
MotR = {'up': 0, 'down': 1}
ColR = {'green': 0, 'red': 1}

def response(cue_choice, col_choice, dir_choice):
  if cue_choice in list([cuelist[0], cuelist[1]]):
    # print('Task:Motion')
    if dir_choice > 5:
      r = MotR['up']
    elif dir_choice < (-5):
      r = MotR['down']
    else:
      r = 2
  else:
    # print('Task:Colour')
    if col_choice > 5:
      r = ColR['red']
    elif col_choice < (-5):
      r = ColR['green']
    else:
      r = 2
  return r


cuelist = ['x', '*', 'o', '^']

def testX(frames,cue_choice, speed_choice, col_choice, dir_choice):
  if cue_choice == 'x':
    xCUE = xCR
  elif cue_choice == '*':
    xCUE = xQ
  elif cue_choice == 'o':
    xCUE = xCI
  else:
    xCUE = xT

  fp = torch.zeros(int(0.5*frames), 3, 128, 128)
  for i in range(int(0.5*frames)):
    fp[i, :, :, :] = xFP
  cue = torch.zeros(frames, 3, 128, 128)
  for j in range(frames):
    cue[j, :, :, :] = xCUE

  stim = genimgs(speed_choice, dir_choice, col_choice)
  test_X = torch.cat((fp, cue, stim), dim=0) 
  output = response(cue_choice, col_choice, dir_choice)

  return test_X, output


# Load Trained CNN (VGG-16)
model = torchvision.models.vgg16(pretrained=True).cuda()
model.eval()
print(model)

return_layers = {
  'classifier.0': 'lr1',
  'classifier.3': 'lr2',
  'classifier.6': 'lr3',
}

# (classifier): Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
# (1): ReLU(inplace=True)
# (2): Dropout(p=0.5, inplace=False)
# (3): Linear(in_features=4096, out_features=4096, bias=True)
# (4): ReLU(inplace=True)
# (5): Dropout(p=0.5, inplace=False)
# (6): Linear(in_features=4096, out_features=1000, bias=True)
# )

N = 30
output_size = 1000
if output_size == 1000:
  out_layer = 'lr3'

for frames in [12, 24, 36, 48, 60, 120, 240]: # default:60
  Nimgs=3*frames

  for index in range(21):
    col_choice = comb[index][0]
    dir_choice = comb[index][1]
    print('Combination No. :', index, 'Combination :', comb[index], 'Colour :', col_choice, 'Direction : ', dir_choice)

    inputs = torch.zeros(2 * 4 * N, int((1.5+3)*frames), output_size)
    OUTcomb = torch.zeros(2 * 4 * N)
    c = 0
    start_time = time.time()
    for speed_choice in ['slow', 'fast']:
      for cue_choice in cuelist:
        print('Speed:', speed_choice)
        print('Cue:', cue_choice)
        for i in range(N):
          test_X, output = testX(frames,cue_choice, speed_choice, col_choice, dir_choice)
          OUTcomb[c] = output 
          testset = TensorDataset(test_X)
          testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                   shuffle=False, num_workers=2)
          del output
          del test_X

          with torch.no_grad():
            for t, data in enumerate(testloader):
              image, = data
              mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
              activations, model_output = mid_getter(image.cuda())
              inputs[c, t, :] = activations[out_layer].squeeze().cpu()
              del image
              del data

            c += 1
    print(inputs.shape) #[240, 270, 400]
    print(OUTcomb.shape) #[240]
    torch.save(inputs, f'LSTMData/LSTMTrainX_{index}_{output_size}_{frames}')
    torch.save(OUTcomb, f'LSTMData/LSTMTrainY_{index}_{output_size}_{frames}')
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


N = 10
for frames in [12, 24, 36, 48, 60, 120, 240]:
  Npoints = 400
  Nimgs = 3 * frames
  for index in range(21):
    col_choice = comb[index][0]
    dir_choice = comb[index][1]
    print('Combination No. :', index, 'Combination :', comb[index], 'Colour :', col_choice, 'Direction : ', dir_choice)
    inputs = torch.zeros(2 * 4 * N, int((1.5+3)*frames), output_size)
    OUTcomb = torch.zeros(2 * 4 * N)
    c = 0
    start_time = time.time()
    for speed_choice in ['slow', 'fast']:
      for cue_choice in cuelist:
        print('Speed:', speed_choice)
        print('Cue:', cue_choice)
        for i in range(N):
          test_X, output = testX(frames,cue_choice, speed_choice, col_choice, dir_choice)
          OUTcomb[c] = output 
          testset = TensorDataset(test_X)
          testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                    shuffle=False, num_workers=3)
          del output
          del test_X

          with torch.no_grad():
            for t, data in enumerate(testloader):
              image, = data
              mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
              activations, model_output = mid_getter(image.cuda())
              inputs[c, t, :] = activations[out_layer].squeeze().cpu()
              torch.cuda.empty_cache()
              del image
              del data

            c += 1
    print(index)
    print(inputs.shape)
    print(OUTcomb.shape) 
    print(f'frames:{frames}')
    torch.save(inputs, f'LSTMData/LSTMTestX_{index}_{output_size}_{frames}')
    torch.save(OUTcomb, f'LSTMData/LSTMTestY_{index}_{output_size}_{frames}')
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))