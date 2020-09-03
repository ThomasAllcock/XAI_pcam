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
from PIL import Image 
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score
import h5py
import shutil

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
    
#%% Image Load Class

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
    
#%% function to show images from tensors

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

#%% save and load chekpoint functions

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)

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

#%% Model function

def train_model(model, criterion, optimizer, scheduler, results,  ckp_dir, best_model_dir, 
                start_epoch=0, num_epochs=25, start_acc=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = start_acc

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print('Epoch {}/{}'.format(epoch, start_epoch+num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # max output
                    loss = criterion(outputs,labels) # Cross-Entropy Loss
                    #preds = (outputs.data > 0.5) * 1 # sigmoid output
                    #loss = criterion(outputs, labels.unsqueeze(1).float()) # sigmoid and BCELoss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds.squeeze(1) == labels.data)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc.item()
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc.item()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    #best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss
                        }
                    save_ckp(checkpoint, True, ckp_dir, best_model_dir)
                else:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss
                        }
                    save_ckp(checkpoint, False, ckp_dir, best_model_dir)

        print()
        results = results.append({results.columns[0] : train_loss ,
                                  results.columns[1] : train_acc  ,
                                  results.columns[2] : val_loss   ,
                                  results.columns[3] : val_acc   },
                                  ignore_index=True)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, results

#%% Visualise_model

def visualise_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#%% Plot Loss and Acc from results dataframe

def model_graph(p1, p2, results, ylab):
    plt.figure() # plot loss at each epoch
    plt.plot(range(1,len(results)+1),results[p1], 'r')
    plt.plot(range(1,len(results)+1),results[p2], 'b')
    plt.xlabel('Epoch')
    plt.ylabel(ylab)
    plt.legend([p1, p2])
    plt.ylim([0,1])

#%% Model accuracy function

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
              total += label.size(0)
              correct += (preds == label).sum().item()
              pred.append(preds)
              lab.append(label)
              outs.append(outputs.data)
    return correct / total, torch.cat(lab), torch.cat(pred), torch.cat(outs)

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
    
    
#%% Load Training, Validation and Test Images
# provide paths to the PCAM training, validation and test sets
metastasis      = read_h5("pcam_dataset\camelyonpatch_level_2_split_train_y.h5")
patchs          = read_h5("pcam_dataset\camelyonpatch_level_2_split_train_x.h5")
metastasis_val  = read_h5("pcam_dataset\camelyonpatch_level_2_split_valid_y.h5")
patchs_val      = read_h5("pcam_dataset\camelyonpatch_level_2_split_valid_x.h5")
metastasis_test = read_h5("pcam_dataset\camelyonpatch_level_2_split_test_y.h5")
patchs_test     = read_h5("pcam_dataset\camelyonpatch_level_2_split_test_x.h5")

#%% Create Dictionaries and DataFrames from training, validation and test images

#norm_patchs = [norm_patch/255 for norm_patch in patchs]
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

#%% Testing Transforms & imshow func

# Get a batch of training data
inputs, classes_1 = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[classes[x] for x in classes_1])


#%% GPU Device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%% Define Model, loss function, optimizer and lr scheduler

model_name = 'VGG11_soft' #  give model a name 

#model_base = models.googlenet(pretrained=True)
#model_base = models.resnet34(pretrained=True)
model_base = models.vgg11(pretrained=True)


#for param in model_base.parameters(): # Freeze Model Layers
#    param.requires_grad = False

#num_ftrs = model_base.fc.in_features

model_base.classifier[-1] = nn.Sequential( # Use with VGG
    nn.Linear(4096, 2))

#model_base.fc = nn.Linear(num_ftrs, 2) # Use with ResNet and GoogleNet

model = model_base.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # changing learning rate every 2 epochs

total_learn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% Create results dataframe

results = pd.DataFrame(columns = ['Train_loss', 'Train_acc',
                                  'Val_loss'  , 'Val_acc']) # create empty results array

#results = pd.read_pickle('models/ResNet18/training/results.pkl') # load previous results if you need to retrain

start_epoch = 0
start_acc = 100 # This is actually the starting Loss value NOT accuracy

#%% Run Model

ckp_dir = './models/'+model_name+'/checkpoint' # checkpoint directory path

best_model_dir = './models/'+model_name+'/best_model' # best model directory path

model, results = train_model(model, criterion, optimizer, exp_lr_scheduler,\
                       results, ckp_dir=ckp_dir, best_model_dir=best_model_dir,
                       start_epoch=start_epoch, num_epochs=4, start_acc=start_acc)

model, optimizer, start_epoch, start_acc = load_ckp(ckp_dir+'/checkpoint.pt', 
                                                    model, optimizer) # Load last Checkpoint

results.to_pickle('models/'+model_name+'/training/results.pkl') # save results to pickle

#%% Load model checkpoint to continue training

model, optimizer, start_epoch, start_acc = load_ckp(ckp_dir+'/checkpoint.pt', model_base, optimizer) # Load last Checkpoint
results = pd.read_pickle('models/'+model_name+'/training/results.pkl') # load previous results

model, optimizer, start_epoch, start_acc = load_ckp(best_model_dir+'/best_model.pt', model_base, optimizer) # Load Best Model
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#%% Plot accuracy or loss per epoch

model_graph('Val_acc', 'Train_acc', results, 'acc')

#%% visualise model prediction

visualise_model(model)

#%% Model accuracy
          
acc, label, preds, outputs = modelaccuracy(model, val_loader, device=device)

print('Accuracy of the network on the images: %d %%' % (100 * acc))  