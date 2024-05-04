import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from torch_geometric.loader import DataLoader
import torch
from In_memory_dataset_whole_IUPAC import MyOwnDataset
from model.ablation_MLP import Net
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
    smile_token_length = args.token_length_smile
    iupac_token_length = args.token_length_iupac

    # define the log directory
    tensorboard_log_dir = "log/ablation/" + args.model_name + "/" + args.dataset + "/epoch_" + str(args.epochs) + \
                          "/batchsize_" + str(args.batch_size)  + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(tensorboard_log_dir, filename_suffix=tensorboard_log_dir.replace("/", "_")[20:])
    # tensorboard --logdir logs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch.nn.functional as F
    tokenizer_smile = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)

    """ data set """

    our_dataset = MyOwnDataset(root="./drug_data/")

    train_size = int(0.8 * len(our_dataset))
    test_size = len(our_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(our_dataset, [train_size, test_size])

    print("size of dataset", len(train_dataset))

    """ data loader """
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True)
    test_loader= DataLoader(test_dataset, batch_size=train_batch_size,shuffle=True)
    """ load model """
    model = Net(number_of_task).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()

    """ train """
    train_losses = []
    train_acces = []


    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0

        model.train()
        for i, (data) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            smiles = data.smiles
            input_smile = tokenizer_smile.batch_encode_plus(smiles, truncation=True, pad_to_max_length=True, return_tensors="pt")
            input_smile = input_smile.to(device)
            outputs = model(data, input_smile)

            data.y = data.y.view(len(data), number_of_task)
            loss = criterion(outputs, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            outputs = outputs.view(-1)
            preds=torch.where(outputs>0.5,1,0)
            data.y=data.y.view(-1)
            num_correct = (preds == data.y).sum().item()
            acc = num_correct / data.y.shape[0]
            train_acc += acc

        writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Train/Acc', train_acc / len(train_loader), epoch)

        eval_loss = 0
        eval_acc = 0

        """ evaluate """
        model.eval()

        preds = []
        trues = []
        belief_scores = []
        # tp = 0
        # tn = 0
        # fp = 0
        # fn = 0

        for i, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                data = data.to(device)
                smiles = data.smiles
                input_smile = tokenizer_smile.batch_encode_plus(smiles, truncation=True, pad_to_max_length=True, return_tensors="pt")
                input_smile = input_smile.to(device)
                outputs = model(data, input_smile)

                data.y = data.y.view(len(data), number_of_task)
                loss = criterion(outputs, data.y)
                eval_loss += loss

                outputs = outputs.view(-1)
                data.y = data.y.view(-1)
                pred=torch.where(outputs>0.5,1,0)
                num_correct = (pred == data.y).sum().item()
                acc = num_correct / data.y.shape[0]
                eval_acc += acc

                labels=data.y
                belief_score = outputs
                pred = pred.cpu().tolist()
                true = labels.cpu().tolist()
                belief_score = belief_score.cpu().tolist()
                preds.extend(pred)
                trues.extend(true)
                belief_scores.extend(belief_score)
        print("roc_auc_score", roc_auc_score(trues, belief_scores))
        trues = np.array(trues).reshape(-1, number_of_task).T
        belief_scores = np.array(belief_scores).reshape(-1, number_of_task).T
        roc_auc_score_list = []
        for i in range(number_of_task):
            temp_roc_auc_score = roc_auc_score(trues[i].tolist(), belief_scores[i].tolist())
            roc_auc_score_list.append(temp_roc_auc_score)

        print("roc_auc_score", sum(roc_auc_score_list) / number_of_task)
        # print("AUPR", average_precision_score(trues, belief_scores))
        writer.add_scalar('Test/Loss', eval_loss / len(test_loader), epoch)
        writer.add_scalar('Test/Acc', eval_acc / len(test_loader), epoch)
        writer.add_scalar('Test/AUC_ROC', sum(roc_auc_score_list) / number_of_task, epoch)

        print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f},'
              'Test Loss:{:.4f},Test Acc:{:.4f}'
              .format(epoch, train_loss / len(train_loader),
                      train_acc / len(train_loader),
                      eval_loss / len(test_loader),
                      eval_acc / len(test_loader)))

        stop_time = time.time()

        print("time is:{:.4f}s".format(stop_time-start_time))


    trues = trues[0]
    image = sd.plot_confusion_matrix_image(preds, trues, ["Class 0", "Class 1"])
    # Log the confusion matrix image to TensorBoard
    writer.add_image("Confusion Matrix", image.permute(2, 0, 1))
    writer.close()

    # sd.save_model(epoch, model.state_dict(), optimizer.state_dict(), loss, tensorboard_log_dir)

