import numpy as np
import torch
import os

from torch.utils.data import TensorDataset, DataLoader
#from braindecode.models import EEGNetv1, EEGNetv4
#from models.EEGNetv1 import EEGNetv1
#from models.LSTM import LSTM
#from sklearn.metrics import accuracy_score, roc_auc_score
from models.EEGNet import EEGNet
from tqdm import tqdm
from models.FC import FC, FC_ELU, FC_ELU_Dropout
from utils.alg_utils import cross_entropy_with_probs, soft_cross_entropy_loss

import random
import sys
import time


def nn_fixepoch(
        model,
        learning_rate,
        num_iterations,
        metrics,
        cuda,
        cuda_device_id,
        seed,
        dataset,
        model_name,
        test_subj_id,
        label_probs,
        valid_percentage,
        train_x,
        train_y,
        test_x,
        test_y,
        #loss_weights
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    if valid_percentage != None and valid_percentage > 0:
        valid_indices = []
        for i in range(len(train_x) // 100):
            indices = np.arange(valid_percentage) + 100 * i
            valid_indices.append(indices)
        valid_indices = np.concatenate(valid_indices)
        valid_x = train_x[valid_indices]
        valid_y = train_y[valid_indices]
        train_x = np.delete(train_x, valid_indices, 0)
        train_y = np.delete(train_y, valid_indices, 0)

        print('train_x.shape, train_y.shape, valid_x.shape, valid_y.shape:', train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    if label_probs:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=64)

    if valid_percentage != None and valid_percentage > 0:
        if label_probs:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y).to(torch.float32)
        else:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y.reshape(-1,)).to(torch.long)
        if model_name == 'EEGNet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(3).permute(0, 3, 1, 2)
        valid_dataset = TensorDataset(tensor_valid_x, tensor_valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=64)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        #if loss_weights is not None:
        #    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
        #else:
        #    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

        if valid_percentage != None and valid_percentage > 0 and (epoch + 1) % 5 == 0:
            model.eval()
            total_loss_eval = 0
            cnt_eval = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x)
                    loss_eval = criterion(outputs, y)
                    total_loss_eval += loss_eval
                    cnt_eval += 1
                eval_loss = np.round(total_loss_eval.cpu().item() / cnt_eval, 5)
                print('valid loss:', eval_loss)

                if eval_loss < best_valid_loss:
                    best_valid_loss = eval_loss
                    valid_cnter = 0
                    torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) +
                               '_best.ckpt')
                else:
                    valid_cnter += 1
                if valid_cnter == 3:
                    stop = True
            model.train()

        #if (epoch + 1) % 100 == 0:
        #    torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) +
        #               '_epoch_' + str(epoch + 1) + '.ckpt')
        #    print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.item())
            y_pred.append(predicted.item())
        score = np.round(metrics(y_true, y_pred), 5)
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


def nn_fixepoch_middlecat(
        model,
        learning_rate,
        num_iterations,
        metrics,
        cuda,
        cuda_device_id,
        seed,
        dataset,
        model_name,
        test_subj_id,
        label_probs,
        valid_percentage,
        train_x,
        train_y,
        test_x,
        test_y,
        middle_feature_train_x,
        middle_feature_test_x,
        feature_total,
        class_out,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    if valid_percentage != None and valid_percentage > 0:
        valid_indices = []
        for i in range(len(train_x) // 100):
            indices = np.arange(valid_percentage) + 100 * i
            valid_indices.append(indices)
        valid_indices = np.concatenate(valid_indices)
        valid_x = train_x[valid_indices]
        valid_y = train_y[valid_indices]
        train_x = np.delete(train_x, valid_indices, 0)
        train_y = np.delete(train_y, valid_indices, 0)

        print('train_x.shape, train_y.shape, valid_x.shape, valid_y.shape:', train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    if label_probs:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=64)

    if valid_percentage != None and valid_percentage > 0:
        if label_probs:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y).to(torch.float32)
        else:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y.reshape(-1,)).to(torch.long)
        if model_name == 'EEGNet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(3).permute(0, 3, 1, 2)
        valid_dataset = TensorDataset(tensor_valid_x, tensor_valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=64)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()

    tensor_middle_feature_train_x = torch.from_numpy(middle_feature_train_x).to(torch.float32).to(device)
    tensor_middle_feature_test_x = torch.from_numpy(middle_feature_test_x).to(torch.float32).to(device)

    #tensor_middle_feature_train_x.requires_grad = True

    model.to(device)
    FC1 = FC_ELU(nn_in=middle_feature_train_x.shape[1], nn_out=middle_feature_train_x.shape[1]).to(device)
    FC2 = FC_ELU(nn_in=(feature_total - middle_feature_train_x.shape[1]), nn_out=32).to(device)
    FC3 = FC(nn_in=(32 + middle_feature_train_x.shape[1]), nn_out=class_out).to(device)

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            feature_x = tensor_middle_feature_train_x[i * 64: (i+1) * 64, :]
            feature_x = FC1(feature_x)
            x = x.to(device)
            y = y.to(device)
            deep_feature_x = model(x)
            deep_feature_x = FC2(deep_feature_x)
            '''
            print('deep_feature_x')
            print(deep_feature_x)
            print('feature_x')
            print(feature_x)
            input('')
            '''
            catted = torch.cat((deep_feature_x, feature_x), 1)
            outputs = FC3(catted)
            loss = criterion(outputs, y)
            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))
        '''
        # TODO
        if valid_percentage != None and valid_percentage > 0 and (epoch + 1) % 5 == 0:
            model.eval()
            total_loss_eval = 0
            cnt_eval = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x)
                    loss_eval = criterion(outputs, y)
                    total_loss_eval += loss_eval
                    cnt_eval += 1
                eval_loss = np.round(total_loss_eval.cpu().item() / cnt_eval, 5)
                print('valid loss:', eval_loss)

                if eval_loss < best_valid_loss:
                    best_valid_loss = eval_loss
                    valid_cnter = 0
                    torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) +
                               '_best.ckpt')
                else:
                    valid_cnter += 1
                if valid_cnter == 3:
                    stop = True
            model.train()
        '''
        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/middlefusion_' + model_name + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '.ckpt')
            torch.save(FC1, './runs/' + str(dataset) + '/middlefusion_' + 'FC1' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '.ckpt')
            torch.save(FC2, './runs/' + str(dataset) + '/middlefusion_' + 'FC2' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '.ckpt')
            torch.save(FC3, './runs/' + str(dataset) + '/middlefusion_' + 'FC3' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break
    '''
    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) + '_best.ckpt')
    '''
    model.eval()
    FC1.eval()
    FC2.eval()
    FC3.eval()
    y_true = []
    y_pred = []
    cnt = 0
    with torch.no_grad():
        for x, y in test_loader:
            feature_x = tensor_middle_feature_test_x[cnt, :].unsqueeze_(0)
            cnt += 1
            feature_x = FC1(feature_x)
            x = x.to(device)
            y = y.to(device)
            deep_feature_x = model(x)
            deep_feature_x = FC2(deep_feature_x)
            catted = torch.cat((deep_feature_x, feature_x), 1)
            outputs = FC3(catted)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.item())
            y_pred.append(predicted.item())
        print('evaliation metrics')
        score = np.round(metrics(y_true, y_pred), 5)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


def nn_fixepoch_doubleinput(
        model,
        learning_rate,
        num_iterations,
        metrics,
        cuda,
        cuda_device_id,
        seed,
        dataset,
        model_name,
        test_subj_id,
        label_probs,
        valid_percentage,
        train_x1,
        train_x2,
        train_y,
        test_x1,
        test_x2,
        test_y,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    if label_probs:
        tensor_train_x1, tensor_train_x2, tensor_train_y = torch.from_numpy(train_x1).to(
            torch.float32), torch.from_numpy(train_x2).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        tensor_train_x1, tensor_train_x2, tensor_train_y = torch.from_numpy(train_x1).to(
            torch.float32), torch.from_numpy(train_x2).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        tensor_train_x1, tensor_train_x2 = tensor_train_x1.unsqueeze_(3).permute(0, 3, 1, 2), tensor_train_x2.unsqueeze_(3).permute(0, 3, 1, 2)
    train_dataset1, train_dataset2 = TensorDataset(tensor_train_x1, tensor_train_y), TensorDataset(tensor_train_x2, tensor_train_y)
    train_loader1, train_loader2 = DataLoader(train_dataset1, batch_size=64), DataLoader(train_dataset2, batch_size=64)

    tensor_test_x1, tensor_test_x2, tensor_test_y = torch.from_numpy(test_x1).to(
        torch.float32), torch.from_numpy(test_x2).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x1, tensor_test_x2 = tensor_test_x1.unsqueeze_(3).permute(0, 3, 1, 2), tensor_test_x2.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset1, test_dataset2 = TensorDataset(tensor_test_x1, tensor_test_y), TensorDataset(tensor_test_x2, tensor_test_y)
    test_loader1, test_loader2 = DataLoader(test_dataset1), DataLoader(test_dataset2)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for a, b in zip(train_loader1, train_loader2):
            # Forward pass
            (x1, y) = a
            (x2, _) = b
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            x = [x1, x2]
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))


        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (x1, y), (x2, _) in zip(test_loader1, test_loader2):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            x = [x1, x2]
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.item())
            y_pred.append(predicted.item())
        score = np.round(metrics(y_true, y_pred), 5)
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


if __name__ == '__main__':
    model_name = 'EEGNet'
    dataset = 'BNCI2014-001'
    avg_arr = []
    for i in range(1, 10):
        out_arr = []
        for s in range(0, 1):
            print('subj',i, 'seed', s)
            out = nn_fixepoch(
                test_subj=i,
                learning_rate=0.001,
                num_iterations=200,
                cuda=False,
                seed=s,
                test=False,
                test_path='./runs/MI2/EEGNetMI2seed' + str(s) +'_pretrain_model_test_subj_' + str(i) + '_epoch100.pt',
                dataset=dataset,
                model_name=model_name,
            )
            print(round(out, 5))
            out_arr.append(round(out, 5))
        print(out_arr)
        avg_arr.append(np.average(out_arr))
    print(dataset, model_name)
    print(avg_arr)
    print(round(np.average(avg_arr), 5))
