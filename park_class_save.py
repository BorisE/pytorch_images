
#%env CUDA_VISIBLE_DEVICES=2

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

from datetime import datetime
start_time = datetime.now()

################################################################
# init variables

# Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}


train_data_loader, valid_data_loader, test_data_loader = None, None, None
train_data_size, valid_data_size, test_data_size = None, None, None
loss_func, optimizer = None, None
history = None, None
best_epoch = 29


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("Device: " + str(device))

dataset = 'dataset_park'

# Load the Data
# Set train and valid directory paths
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')
test_directory = os.path.join(dataset, 'test')

 # Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory),
    'valid': datasets.ImageFolder(root=valid_directory),
    'test': datasets.ImageFolder(root=test_directory)
}
# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)

print('Duration: {}'.format(datetime.now() - start_time))

def RunTraining():

    global trained_model, train_data_loader, valid_data_loader, test_data_loader
    global best_epoch, history

    ################################################################
    # Data Loading
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }


    # Batch size
    bs = 32
    # Number of classes
    num_classes = len(os.listdir(valid_directory))  #10#2#257
    print("Number of classes " + str(num_classes))

    #num_classes = 10
    #print("Limiting it to " + str(num_classes))

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader  = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader  = DataLoader(data['valid'], batch_size=bs, shuffle=True)
    test_data_loader  = DataLoader(data['test'], batch_size=bs, shuffle=True)

    # Print the train, validation and test set data sizes
    train_data_size, valid_data_size, test_data_size
    print(train_data_size, valid_data_size, test_data_size)



    ################################################################
    # Transfer Learning

    # Load pretrained ResNet50 Model
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = resnet50.to(device)

    # Freeze model parameters
    for param in resnet50.parameters():
        param.requires_grad = False

    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.fc.in_features

    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    # Convert model to be used on GPU
    resnet50 = resnet50.to(device)

    # Define Optimizer and Loss Function
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    # Print the model to be trained
    #summary(resnet50, input_size=(3, 224, 224), batch_size=bs, device='cuda')

    # Train the model for 25 epochs
    num_epochs = 30
    trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

    torch.save(history, dataset + "/" + dataset+'_history.pt')


################################################################
# Training
def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            # Compute loss
            loss = loss_criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))



        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
        
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/float(train_data_size)
        
        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/float(valid_data_size)
        
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()
        
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

        # Save if the model has best accuracy till now
        torch.save(model, dataset + "/" + dataset+'_model_'+str(epoch)+'.pt')

    return model, history, best_epoch



def ShowPlots():
    history = np.array(history)

    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.savefig(dataset+'_loss_curve.png')
    plt.show()


    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.savefig(dataset+'_accuracy_curve.png')
    plt.show()


def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))

################################################################
# Inference
def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    #print('Predict start. Duration: {}'.format(datetime.now() - start_time))
    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    #plt.imshow(test_image)

    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    #print('Predict test_image_tensor. Duration: {}'.format(datetime.now() - start_time))

    with torch.no_grad():
        model.eval()
        #print('Predict model.eval(). Duration: {}'.format(datetime.now() - start_time))

        # Model outputs log probabilities
        out = model(test_image_tensor)
        #print('Predict model(test_image_tensor). Duration: {}'.format(datetime.now() - start_time))
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]

        print("*** Image: " + test_image_name + " ***") 
        print("The most probable it is: " + cls + " [" + str(score) + "]") 
        for i in range(2):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])


################################################################
# Retraing the model

#RunTraining()

#print('Duration: {}'.format(datetime.now() - start_time))

#ShowPlots()

print('Duration: {}'.format(datetime.now() - start_time))

################################################################
# Test a particular model on a test image

model = torch.load(dataset + "/" + "{}_model_{}.pt".format(dataset, best_epoch))
print('Duration: {}'.format(datetime.now() - start_time))

predict(model, 'AstroCam_1.jpg')
print('Duration: {}'.format(datetime.now() - start_time))

predict(model, 'AstroCam_2.jpg')

predict(model, 'AstroCam_3.jpg')

predict(model, 'AstroCam_4.jpg')

predict(model, 'AstroCam_5.jpg')

predict(model, 'AstroCam_6.jpg')

predict(model, 'AstroCam_7.jpg')

predict(model, 'AstroCam_8.jpg')

print('Duration: {}'.format(datetime.now() - start_time))

# Load Data from folders
if (loss_func):
    computeTestSetAccuracy(model, loss_func)