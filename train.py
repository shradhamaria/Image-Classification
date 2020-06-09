import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture: vgg16, resnet18, densenet201')
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Save directory for checkpoints')
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate')
    parser.add_argument('--hidden_units', 
                        type=int, default = 500,
                        help='Hidden units')
    parser.add_argument('--epochs', 
                        type=int, default = 1,
                        help='Number of epochs for training')
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
    # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    
    return device

def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        if architecture == 'densenet201':
            model = models.densenet201(pretrained=True)
            model.name = 'densenet201'
            print("Network architecture specified as densenet201.")
        elif architecture == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.name = 'resnet18'
            print("Network architecture specified as resnet18.")
        else:
            raise RuntimeError("Unknown model")
            
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False 
    return model

def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        print("Number of Hidden Layers specificed as 500 .")
    
    # Find Input Layers
    if model.name == "vgg16":
        input_features = model.classifier[0].in_features
    elif model.name == "densenet201":
        input_features = model.classifier.in_features
    else:
        input_features = model.fc.in_features
        
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(model, trainloader, testloader, device, criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 5
        print("Number of Epochs specificed as 5.")    
 
    print("Training process initializing .....\n")
    
    model.train()
    
    # Train Model
    for e in range(epochs):
        running_loss = 0
         
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, testloader, criterion,device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    return model

def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            if Model.name == "resnet18":
                checkpoint = {'architecture': Model.name,
                          'classifier': Model.fc,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            else:
                checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else: 
            print("Directory not found, model will not be saved.")

            
def main():
     
    args = arg_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    if args.arch == ('resnet18'):
        model.fc = initial_classifier(model,hidden_units=args.hidden_units)
        
    else:
        model.classifier = initial_classifier(model,hidden_units=args.hidden_units)

    device = check_gpu(gpu_arg=args.gpu);
    
    model.to(device);
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    if args.arch == ('resnet18'):
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    
    print_every = 30
    steps = 0
    

    
    # Train the classifier layers
    trained_model = network_trainer(model, trainloader, validloader,device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data)

if __name__ == '__main__': main()