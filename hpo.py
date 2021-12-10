#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
from smdebug.pytorch import get_hook

import argparse

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    if hook:
        hook.set_mode(smd.modes.EVAL)
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device)
            target=target.to(device)
            outputs=model(data)
            loss=criterion(outputs,target)
            _,preds=torch.max(outputs,1)
            test_loss+=loss.item() * data.size(0)
            correct+=torch.sum(preds==target.data).item()
    total_loss=test_loss/len(test_loader)
    total_acc=correct/len(test_loader)
    
     #TODO - what happens to total_loss / total_acc? Taken from 'finetune a cnn' exercise
    
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    
    #Adapted from 2 other exercises
    #TODO - does .item() work as expected? should test independently
    
def train(model, train_loader, criterion, optimizer, device, hook, batch_size, epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    for epoch in range(epochs):
        if hook:
            hook.set_mode(smd.modes.TRAIN)
        model.train()
        running_loss=0
        correct=0
        total_trained=0
        
        for data,target in train_loader:
            data=data.to(device)
            target=target.to(device)
            optimizer.zero_grad()
            pred=model(data)
            loss=criterion(pred,target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1,keepdim=True) #What does this do
            correct+=pred.eq(target.view_as(pred)).sum().item()
            total_trained+=batch_size
        print(f"Epoch{epoch}: Loss {running_loss/(batch_size*len(train_loader))}, Accuracy {100*(correct/(batch_size*len(train_loader)))}%")
        
    return model
     
def net(device):
    #Create model - method could also be called model() to fit with previous lectures
    #Sticking with resnet18 due to model size/training length
    model=models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    num_features=model.fc.in_features
    model.fc=nn.Sequential(nn.Linear(num_features,133)) #133 dog breeds used
    model.fc2=nn.Sequential(nn.Linear(133,133)) #133 dog breeds used - second FC layer
    model.to(device)
    return model

def create_data_loaders(batch_size, train_folder, test_folder):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])
    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()])
    
    train_dataset=torchvision.datasets.ImageFolder(root=train_folder,transform=training_transform)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    test_dataset=torchvision.datasets.ImageFolder(root=test_folder,transform=testing_transform)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    
    return train_loader,test_loader

def main(args):
    
    # Some invalid image data in dataset, requires this solution to load, possibly has bad image data
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net(device)
    
#     hook=smd.Hook.create_from_json_file() //This is the code in the module for learning Debugger, yet the exercise does not use this??
#     hook.register_hook(model) //This is the code in the module for learning Debugger, yet the exercise does not use this??
    
    import os
    
    # Paths are specified in fit() method call
    train_folder=os.environ['SM_CHANNEL_TRAIN']
    val_folder=os.environ['SM_CHANNEL_VAL']
    test_folder=os.environ['SM_CHANNEL_TEST']
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    hook=get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(loss_citerion)
    
    
    batch_size=args.batch_size
    epochs=args.epochs
        
    train_loader,test_loader=create_data_loaders(batch_size,train_folder,test_folder)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer, device, hook, batch_size, epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model - Amazon returns the model by default, why save? 
    '''
    
#     path="./pytorchmodel.pth"
#     torch.save(model, path)
    
#     !aws s3 sync "./pytorchmodel.pth" "s3://sagemaker-studio-blhsxxxbf57/project3/torchmodel/"
    
#     path="s3://sagemaker-studio-blhsxxxbf57/project3/torchmodel/"

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training and testing (default: 64)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="input number of epochs (default: 5)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)"
    )
    
    args=parser.parse_args()
    
    main(args)
    

