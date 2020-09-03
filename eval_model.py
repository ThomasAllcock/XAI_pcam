import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import utils, models, datasets
from  torch.utils.data import Dataset
import time
import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score
#%% Load GPU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Load Model

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_loss']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], best_acc

#%% Manual model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.sig = nn.Sigmoid()
        
        #x = torch.randn(3,90,90).view(-1,3,90,90)
        #self._to_linear = None
        #self.convs(x)

        self.fc1   = nn.Linear(4608, 512) 
        self.fc2   = nn.Linear(512, 1)
        
    def convs(self,x): 
        # block 1
        x = F.dropout(F.relu(self.conv1(x)),p=0.3)
        x = F.dropout(F.relu(self.conv2(x)),p=0.3)
        x = F.dropout(F.relu(self.conv2(x)),p=0.3)
        x = self.pool(x)
        # block 2
        x = F.dropout(F.relu(self.conv3(x)),p=0.3)
        x = F.dropout(F.relu(self.conv4(x)),p=0.3)
        x = F.dropout(F.relu(self.conv4(x)),p=0.3)
        x = self.pool(x)
        # block 3
        x = F.dropout(F.relu(self.conv5(x)),p=0.3)
        x = F.dropout(F.relu(self.conv6(x)),p=0.3)
        x = F.dropout(F.relu(self.conv6(x)),p=0.3)
        x = self.pool(x)
        
        #if self._to_linear is None: # this automatically works out dimension for fc1
        #    self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        #    print(self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
            
        return x
    
    
    
model = Net().to(device)
model_name = 'Net1'
ckp_dir = 'pcam/models/'+model_name+'/checkpoint'
best_model_dir = 'pcam/models/'+model_name+'/best_model'
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model, optimizer, start_epoch, start_acc = load_ckp(best_model_dir+'/best_model.pt', model, optimizer)
model.eval()

#%% GoogLeNet
# model_ft = models.resnet18(pretrained=True)
model_ft = models.googlenet(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
#model_ft.fc = nn.Sequential(
#    nn.Linear(num_ftrs, 1),
#    nn.Sigmoid()
#)
model_googlenet = model_ft.to(device)

model_name = 'GoogLeNet_soft'
ckp_dir = 'pcam/models/'+model_name+'/checkpoint'
best_model_dir = 'pcam/models/'+model_name+'/best_model'
optimizer = optim.SGD(model_googlenet.parameters(), lr=0.001, momentum=0.9)
model_googlenet, optimizer, start_epoch, start_acc = load_ckp(best_model_dir+'/best_model.pt', model_googlenet, optimizer)
model_googlenet.eval()

results_googlenet = pd.read_pickle('pcam/models/'+model_name+'/training/results.pkl') # load previous results

#%% ResNet34
model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Sequential(
#    nn.Linear(num_ftrs, 1),
#    nn.Sigmoid()
#)
model_ft.fc = nn.Linear(num_ftrs, 2)
model_resnet34 = model_ft.to(device)
model_name = 'ResNet34_soft'
ckp_dir = 'pcam/models/'+model_name+'/checkpoint'
best_model_dir = 'pcam/models/'+model_name+'/best_model'
optimizer = optim.SGD(model_resnet34.parameters(), lr=0.001, momentum=0.9)
model_resnet34, optimizer, start_epoch, start_acc = load_ckp(best_model_dir+'/best_model.pt', model_resnet34, optimizer)
model_resnet34.eval()

results_resnet34 = pd.read_pickle('pcam/models/'+model_name+'/training/results.pkl') # load previous results

#%%VGG_11
model_ft= models.vgg11(pretrained=True)
model_ft.classifier[-1] = nn.Sequential(
    nn.Linear(4096, 2))
model_vgg = model_ft.to(device)
model_name = 'VGG11_soft'
ckp_dir = 'pcam/models/'+model_name+'/checkpoint'
best_model_dir = 'pcam/models/'+model_name+'/best_model'
optimizer = optim.SGD(model_vgg.parameters(), lr=0.001, momentum=0.9)
model_vgg, optimizer, start_epoch, start_acc = load_ckp(best_model_dir+'/best_model.pt', model_vgg, optimizer)

results_vgg = pd.read_pickle('pcam/models/'+model_name+'/training/results.pkl')

#%% Plot Loss and Acc from results dataframe

def model_graph(p1, p2, results, ylab):
    plt.figure() # plot loss at each epoch
    plt.plot(range(1,len(results)+1),results[p1], 'r')
    plt.plot(range(1,len(results)+1),results[p2], 'b')
    plt.xlabel('Epoch')
    plt.ylabel(ylab)
    plt.legend([p1, p2])
    plt.ylim([0,1])

#%%
model_graph('Val_loss', 'Train_loss', results_googlenet, 'Loss')
model_graph('Val_loss', 'Train_loss', results_resnet34, 'Loss')
model_graph('Val_loss', 'Train_loss', results_vgg, 'Loss')
#%% read_h5
# Reads a .h5 file

def read_h5(filename, n=None):
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
    
        # Get the data
        if n == None:
            return list(f[a_group_key])
        else:
            return list(f[a_group_key][0:n])
    
#%% Image_set
class Image_set(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = Image.fromarray(self.df['patchs'][index].astype('uint8'), 'RGB')
        y = torch.tensor(int(self.df['metastasis'][index]))
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
#%% imshow

def imshow(inp, title=None, alpha=1):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, alpha=alpha)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

#%% Load data

metastasis      = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_train_y.h5",1)
patchs          = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_train_x.h5",1)
metastasis_val  = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_valid_y.h5",1)
patchs_val      = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_valid_x.h5",1)
metastasis_test = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_test_y.h5")
patchs_test     = read_h5("pcam\pcam_dataset\camelyonpatch_level_2_split_test_x.h5")


#%% Create Dictionaries and DataFrames from training, validation and test images

norm_patchs = [norm_patch/255 for norm_patch in patchs]
data_train = {'patchs':patchs,
        'metastasis':metastasis}
data_val   = {'patchs':patchs_val,
        'metastasis':metastasis_val}
data_test  = {'patchs':patchs_test,
        'metastasis':metastasis_test}

data_df_train = pd.DataFrame(data_train, columns=['patchs', 'metastasis'])
data_df_val = pd.DataFrame(data_val, columns=['patchs', 'metastasis'])
data_df_test = pd.DataFrame(data_test, columns=['patchs', 'metastasis'])    

#%% Data transforms

# Transforms needed for ResNet
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#%% Create Data Sets using Image_set class
# Image_set creates a class that combines both x and y data and also applies
# the transforms to the images (x)

train_size = int(len(data_df_train))
val_size = int(len(data_df_val))
test_size = int(len(data_df_test))

dataset_sizes = {'train':train_size,'val': val_size,'test' : test_size}

data_train = Image_set(df=data_df_train.reset_index(drop=True),
                    transform=data_transform)

data_val = Image_set(df=data_df_val.reset_index(drop=True),
                    transform=data_transform)

data_test = Image_set(df=data_df_test.reset_index(drop=True),
                      transform=data_transform)

class_names = ['normal', 'metastasis' ]

classes = np.arange(0,2)


#%% Create Data Loaders

batch_size = 32

train_loader = torch.utils.data.DataLoader(data_train,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=0
                                           )


val_loader = torch.utils.data.DataLoader(data_val,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=0
                                          )

test_loader = torch.utils.data.DataLoader(data_test,
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=0
                                          )

dataloader = {'train':train_loader,
     'val':val_loader,'test':test_loader} # dataloader dictionary for easy access

#%% Check Model accuracy

def modelaccuracy(model, loader, device=None):

    correct = 0
    total = 0
    pred = []
    lab = []
    outs = []
    
    with torch.no_grad():
        
          # Iterate over the val set
          for data in loader:
              images, label = data
              if device is not None:
                  images = images.to(device)
                  label = label.to(device)
              outputs = model(images)
            
              # torch.max is an argmax operation
              _, preds = torch.max(outputs.data,1)
              #preds = (outputs.data > 0.5) * 1 # sigmoid output
              total += label.size(0)
              correct += (preds == label).sum().item()
              #correct += (preds.squeeze(1) == label).sum().item()
              pred.append(preds)
              lab.append(label)
              outs.append(outputs.data)
    return correct / total, torch.cat(lab), torch.cat(pred), torch.cat(outs)

#%% Accuracy

acc,label,preds,outputs = modelaccuracy(model_vgg, test_loader, device=device)
print('Accuracy of the network on the images: %d %%' % (100 * acc))

#%% AUROC Score

outputs2 = torch.softmax(outputs.cpu(),1)

roc_score = roc_auc_score(label.cpu(), outputs2[:,1])

fpr, tpr, thresholds = roc_curve(label.cpu(),  outputs2[:,1])


accuracies = []
for thresh in thresholds:
    preds = (outputs > thresh) * 1
    correct = (preds.squeeze(1) == label).sum().item()
    accuracies.append(correct/len(preds))
    
#%% F1-score

f1 = f1_score(label.cpu(),preds.cpu())    
    
#%% Confusion matrix function

import itertools

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix very prettily.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # Specify the tick marks and axis text
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # The data formatting
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
#%% Show Confusion Matrix

cm = confusion_matrix(label.cpu(), preds.cpu())
plot_confusion_matrix(cm,classes)

#%% Torchray evaluation
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.guided_backprop import guided_backprop
import multiresolutionimageinterface as mir
from torchray.benchmark.pointing_game import PointingGame
from scipy.ndimage import gaussian_filter
from torchvision.utils import save_image
#%% Load Meta Data

reader = mir.MultiResolutionImageReader() # define reader
meta = pd.read_csv('pcam\pcam_dataset\meta\camelyonpatch_level_2_split_test_meta.csv') # meta data
anot_transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]) # annotation transform
annotation_path = 'Cam/Data/cam16/testing/masks/mask_'

#%% Segmentation
data = data_test
model = model_googlenet.to(device)
n = 69
filenum = meta.iloc[n]['wsi'][-3:]
x = meta.iloc[n]['coord_x']
y = meta.iloc[n]['coord_y']

mr_anot = reader.open(annotation_path+filenum+'.tif')
level = 2
ds = mr_anot.getLevelDownsample(level)
dim_x, dim_y = mr_anot.getLevelDimensions(level=level)
anot_patch = mr_anot.getUCharPatch(int(x), int(y), 96, 96, level)
T_anot_patch = anot_transform(Image.fromarray(anot_patch[:,:,0]))
T_anot_patch_binary = (T_anot_patch > 0) 
# Segmentation Mask
imshow(data[n][0].cpu())
plt.imshow(T_anot_patch_binary.squeeze(0), alpha=0.6)
plt.axis('off')
plt.show()


print('Ground-Truth: '+str(data[n][1].item()))
print('Predicted:    '+str(preds[n].item()))
#%% Perturbation
testimage = data[n][0]
im = testimage.unsqueeze(0).to(device)
masks_1, _ = extremal_perturbation(
    model, im, target=preds[n].cpu().item(),
    #reward_func=contrastive_reward,
    variant = 'preserve',
    debug=False,
    areas=[0.15],
)

plt.figure()
imshow(data[n][0].cpu())
plt.imshow(T_anot_patch_binary.squeeze(0), alpha=0.5)
plt.imshow(masks_1.squeeze(0)[0].cpu(),cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()


#%% Grad-Cam 
testimage = data[n][0]
im = testimage.unsqueeze(0).to(device)
saliency = grad_cam(model, im, saliency_layer='features', target=preds[n].cpu().item()) # inception5b layer4 features
masks_2 = F.upsample(saliency, size=(224,224), mode='bilinear')

imshow(data[n][0].cpu())
plt.imshow(T_anot_patch_binary.squeeze(0), alpha=0.5)
plt.imshow(masks_2.squeeze(0)[0].cpu().detach(),cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()

#%% Guided backprop
testimage = data[n][0]
im = testimage.unsqueeze(0).to(device)
masks_3 = guided_backprop(model, im, target=preds[n].cpu().item())

imshow(data[n][0].cpu())
plt.imshow(masks_3.squeeze(0)[0].cpu().detach(),cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()

#%% Pointing game
point_nums = pd.read_csv('smallpoint_nums.csv')
pg = PointingGame(num_classes=2,tolerance=15)
pg_results = []
for n in point_nums.image_num:
    filenum = meta.iloc[n]['wsi'][-3:]
    x = meta.iloc[n]['coord_x']
    y = meta.iloc[n]['coord_y']
    mr_anot = reader.open(annotation_path+filenum+'.tif')
    if mr_anot is not None: #and preds[n].cpu().item()==1: # for NoneType create patch of just zeros as floats
        # get annotated region
        level = 2
        ds = mr_anot.getLevelDownsample(level)
        dim_x, dim_y = mr_anot.getLevelDimensions(level=level)
        anot_patch = mr_anot.getUCharPatch(int(x), int(y), 96, 96, level)
        T_anot_patch = anot_transform(Image.fromarray(anot_patch[:,:,0]))
        T_anot_patch_binary = (T_anot_patch > 0)
        # get saliency map
        testimage = data[n][0]
        im = testimage.unsqueeze(0).to(device)
        
        mask, _ = extremal_perturbation(model, im, target=label[n].cpu().item(),
                                           #reward_func=contrastive_reward,
                                           variant = 'preserve',
                                           debug=False,
                                           areas=[0.15],
                                           )
        m = mask.squeeze(0)[0].detach().cpu().numpy()
        
        #saliency = grad_cam(model, im, saliency_layer="inception5b", target=label[n].cpu().item())
        #mask = F.upsample(saliency, size=(224,224), mode='bilinear')
        #m = mask.squeeze(0)[0].detach().cpu().numpy()
        # get maximum point(s)
        if mask.max().item() > 0 :
            point = np.where(m==mask.max().item())
        #print(point)
        # evaluate point
            pg_eval = pg.evaluate(T_anot_patch_binary.squeeze(0),[point[1][0],point[0][0]]) # note: rows are y and columns are x
        else: 
            pg_eval = -1
            
        pg_result = (pg_eval+1)/2
        pg_results.append(pg_result)
        #imshow(data[n][0].cpu())
        #plt.imshow(masks_2.squeeze(0)[0].cpu().detach(),cmap='jet', alpha=0.5)
        #plt.imshow(T_anot_patch_binary.squeeze(0), alpha=0.6)
        #plt.show()
        #print('Point Game result for above image: ',pg_eval)
        #print('Ground-Truth: '+str(data[n][1].item()))
        #print('Predicted:    '+str(preds[n].item()))
#%%
match = []
for n in point_nums.image_num:
    match.append((preds[n].cpu().item()==label[n].cpu().item()) * 1)
print(np.sum(match)/len(match))
    

    

#%%
save_image(masks_1.squeeze(0), 'img1.png')
