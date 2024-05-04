from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error
import torch

import shared_method
import shared_method as sd
from In_memory_dataset_whole_IUPAC import MyOwnDataset
from model.model_ablation_regression_GAT import Net
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm




if __name__ == '__main__':
    """ parameters """
    args = sd.parse_input()

    epochs = args.epochs
    train_batch_size = args.batch_size
    number_of_task = args.n_tasks
    training_task = args.dataset
    model_name = args.model_name
    args.model_name = "ablation_regression_GAT"

    # define the log directory, cut from first 20 because it's too long and result in bug
    tensorboard_log_dir = sd.def_log_dir(args)
    writer = SummaryWriter(tensorboard_log_dir, filename_suffix=tensorboard_log_dir.replace("/", "_")[20:])
    # tensorboard --logdir logs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ split data """
    our_dataset = MyOwnDataset(root="drug_data/")
    train_loader, test_loader, train_size, test_size = shared_method.split_data(our_dataset, args.random_seed, args.batch_size)

    """ load model """
    model = Net(number_of_task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    """ train """
    preds = []
    trues = []
    rmse = 0
    min_rmse = 100

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        model.train()
        for i, (data) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            outputs = model(data)

            data.y = data.y.view(len(data), number_of_task)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            preds.extend(outputs.cpu().tolist())
            trues.extend(data.y.cpu().tolist())

        rmse = np.sqrt(mean_squared_error(trues, preds))
        train_loss = criterion(torch.tensor(preds), torch.tensor(trues))
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/rmse', rmse, epoch)


        eval_loss = 0
        model.eval()
        """ evaluate """
        preds = []
        trues = []

        for i, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                data = data.to(device)
                outputs = model(data)

                data.y = data.y.view(len(data), number_of_task)
                loss = criterion(outputs, data.y)

                eval_loss += loss
                preds.extend(outputs.cpu().tolist())
                trues.extend(data.y.cpu().tolist())

        eval_rmse = np.sqrt(mean_squared_error(trues, preds))
        eval_loss = criterion(torch.tensor(preds), torch.tensor(trues))
        writer.add_scalar('Test/Loss', eval_loss, epoch)
        writer.add_scalar('Test/rmse', eval_rmse, epoch)
        if min_rmse > eval_rmse:
            min_rmse = eval_rmse

        print('epoch:{},Train Loss:{:.4f},Train Rmse:{:.4f},'
              'Test Loss:{:.4f},Test Rmse:{:.4f}'
              .format(epoch,
                      train_loss,
                      rmse,
                      eval_loss,
                      eval_rmse))

        stop_time = time.time()

        print("time is:{:.4f}s".format(stop_time-start_time))

    shared_method.add_regression_data_number_scaler(writer, len(test_loader) * args.batch_size)
    print(model_name, args.random_seed, min_rmse)
    writer.close()
    # sd.save_model(epoch, model.state_dict(), optimizer.state_dict(), loss, tensorboard_log_dir)

