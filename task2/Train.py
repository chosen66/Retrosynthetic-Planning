from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import Dataset2, MLP, mkdir_p
from dgllife.utils import EarlyStopping


DATASET_DIR = "../Project for ML/MoleculeEvaluationData/"

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        x, y = batch_data
        y_pred = model.forward(x)
        loss = loss_criterion(y, y_pred)
        total_loss += loss
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()      
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader)), end='', flush=True)

    print('\nepoch %d/%d, training loss: %.4f' % (epoch + 1, args['num_epochs'], total_loss/batch_id))

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            x, y = batch_data
            y_pred = model.forward(x)
            loss = loss_criterion(y_pred, y)
            total_loss += loss
    return total_loss/len(data_loader)

def main(args):
    model_name = 'Task2.pth'
    args['model_path'] = './models/' + model_name
    mkdir_p('./models')                          
    model, loss_criterion = MLP(), nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['schedule_step'])
    stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
    train_data = Dataset2(DATASET_DIR + "train.pkl")
    train_loader = DataLoader(train_data, batch_size=64)
    test_data = Dataset2(DATASET_DIR + "test.pkl")
    test_loader = DataLoader(test_data, batch_size=64)
    test_loss = 10086
    for epoch in range(args['num_epochs']):
        
        early_stop = stopper.step(test_loss, model)
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        test_loss = run_an_eval_epoch(args, model, test_loader, loss_criterion)
        
        scheduler.step()
        print('epoch %d/%d, test loss: %.4f' %  (epoch + 1, args['num_epochs'], test_loss))
        print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))
        if early_stop:
            print ('Early stopped!!')
            break


if __name__ == "__main__":
    parser = ArgumentParser('Task2 training arguements')
    parser.add_argument('-d', '--dataset', default=DATASET_DIR, help='Dataset direction')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=4, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])
    main(args)
        
    
    





