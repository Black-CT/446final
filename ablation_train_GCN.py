import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from torch_geometric.loader import DataLoader
import torch
from In_memory_dataset_whole_IUPAC import MyOwnDataset
from model.ablation_GCN import Net
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizerFast, RobertaTokenizer
from tqdm import tqdm
import datetime
import shared_method as sd




if __name__ == '__main__':
    """ parameters """
    args = sd.parse_input()

    epochs = args.epochs
    train_batch_size = args.batch_size
    number_of_task = args.n_tasks
    training_task = args.dataset
    model_name = args.model_name
    args.model_name = "GCN_model"


    # define the log directory, cut from first 20 because it's too long and result in bug
    tensorboard_log_dir = sd.def_log_dir(args)
    writer = SummaryWriter(tensorboard_log_dir, filename_suffix=tensorboard_log_dir.replace("/", "_")[20:])
    # tensorboard --logdir logs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ split data """
    our_dataset = MyOwnDataset(root="drug_data/")
    train_loader, test_loader, train_size, test_size = sd.split_data(our_dataset, args.random_seed, args.batch_size)

    """ load model """
    model = Net(number_of_task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    """ train """
    train_losses = []
    train_acces = []
    max_auc_roc = 0

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        model.train()

        for i, (data) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            outputs = model(data)

            data.y = data.y.view(len(data), number_of_task)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss * len(data)

            outputs = outputs.view(-1)
            preds=torch.where(outputs>0.5,1,0)
            data.y=data.y.view(-1)
            num_correct = (preds == data.y).sum().item()
            train_acc += num_correct

        writer.add_scalar('Train/Loss', train_loss / train_size, epoch)
        writer.add_scalar('Train/Acc', train_acc / train_size, epoch)



        """ evaluate """
        model.eval()
        eval_loss = 0
        eval_acc = 0
        preds = []
        trues = []
        belief_scores = []

        for i, data in enumerate(tqdm(test_loader)):

            with torch.no_grad():
                data = data.to(device)
                outputs = model(data)

                data.y = data.y.view(len(data), number_of_task)
                loss = criterion(outputs, data.y)
                eval_loss += loss * len(data)

                outputs = outputs.view(-1)
                data.y = data.y.view(-1)
                pred=torch.where(outputs>0.5,1,0)
                num_correct = (pred == data.y).sum().item()
                eval_acc += num_correct

                labels=data.y
                pred = pred.cpu().tolist()
                true = labels.cpu().tolist()
                preds.extend(pred)
                trues.extend(true)

        belief_scores = preds
        trues = np.array(trues).reshape(-1, number_of_task).T
        belief_scores = np.array(belief_scores).reshape(-1, number_of_task).T
        roc_auc_score_list = []
        for i in range(number_of_task):
            temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
            roc_auc_score_list.append(temp_roc_auc_score)

        cur_auc_roc = sum(roc_auc_score_list) / number_of_task
        if max_auc_roc < cur_auc_roc:
            max_auc_roc = cur_auc_roc

        print("roc_auc_score", cur_auc_roc)
        writer.add_scalar('Test/Loss', eval_loss / test_size, epoch)
        writer.add_scalar('Test/Acc', eval_acc / test_size, epoch)
        writer.add_scalar('Test/AUC_ROC', sum(roc_auc_score_list) / number_of_task, epoch)

        print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f},'
              'Test Loss:{:.4f},Test Acc:{:.4f}'
              .format(epoch, train_loss / train_size,
                      train_acc / train_size,
                      eval_loss / test_size,
                      eval_acc / test_size))

        stop_time = time.time()

        print("time is:{:.4f}s".format(stop_time-start_time))
        print(args.model_name, args.random_seed, max_auc_roc)


    trues = trues[0]
    image = sd.plot_confusion_matrix_image(preds, trues, ["Class 0", "Class 1"])

    # Log the confusion matrix image to TensorBoard
    writer.add_image("Confusion Matrix", image.permute(2, 0, 1))
    writer.close()
