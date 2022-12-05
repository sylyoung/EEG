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
from models.FC import FC, FC_ELU, FC_ELU_Dropout, FC_2layer
from models.CNN import Conv1, ConvChannel, ConvFusion, ConvChannelWise
from utils.alg_utils import cross_entropy_with_probs, soft_cross_entropy_loss
from utils.loss import ClassConfusionLoss
from sklearn.metrics.pairwise import cosine_distances

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
        loss_weights=None
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

    batch_size = 32 # 32 256 # TODO

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

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
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

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
            #loss = criterion(outputs, y)

            softmax_out = torch.nn.Softmax(dim=1)(outputs / 2)  # TODO temperature with softmax
            loss = criterion(softmax_out, y)

            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

        if (epoch + 1) % 10 == 0:
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                print('test score:', np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(
                    -1, ).tolist(), 5)[0])
            model.train()

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

        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
        score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


def nn_fixepoch_siamesefusion(
        model_data,
        model_knowledge,
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
        feature_deep_dim,
        class_out,
        ch_num=None,
        loss_weights=None
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

    batch_size = 128 # TODO

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

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
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model_knowledge.to(device)
    model_data.to(device)
    #opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if label_probs:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(middle_feature_train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(middle_feature_train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        feature_tensor_train_x = feature_tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_train_dataset = TensorDataset(feature_tensor_train_x, feature_tensor_train_y)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=batch_size)

    feature_tensor_test_x, feature_tensor_test_y = torch.from_numpy(middle_feature_test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        feature_tensor_test_x = feature_tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_test_dataset = TensorDataset(feature_tensor_test_x, feature_tensor_test_y)
    feature_test_loader = DataLoader(feature_test_dataset, batch_size=batch_size)

    #out_channels = 1

    #FC1 = FC(nn_in=middle_feature_train_x.shape[1], nn_out=middle_feature_train_x.shape[1]).to(device)
    #FC2 = FC(nn_in=feature_deep_dim, nn_out=feature_deep_dim).to(device)
    #Conv = Conv1().to(device)
    #Conv = ConvChannel(in_channels=tensor_feature_train_x.shape[1], out_channels=out_channels, kernel_size=1, bias=False, groups=1).to(device)
    FC3 = FC(nn_in=feature_deep_dim, nn_out=class_out).to(device)
    #FC3 = FC_2layer(nn_in=(middle_feature_train_x.shape[1] + feature_deep_dim), nn_out=class_out).to(device)
    #FC3 = FC(nn_in=(74 * out_channels + feature_deep_dim), nn_out=class_out).to(device)

    #model_fusion = ConvFusion(nn_deep=(74 * out_channels + feature_deep_dim), nn_out=class_out,
    #                in_channels=tensor_feature_train_x.shape[1], out_channels=out_channels,
    #                kernel_size=1, bias=False, groups=1).to(device)


    #all_parameters = list(model.parameters()) + list(FC1.parameters()) + list(FC2.parameters()) + list(FC3.parameters())
    #all_parameters = list(model.parameters()) + list(Conv.parameters()) + list(FC3.parameters())
    all_parameters = list(model_data.parameters()) + list(model_knowledge.parameters()) + list(FC3.parameters())
    #all_parameters = list(model.parameters()) + list(model_fusion.parameters())
    opt = torch.optim.Adam(all_parameters, lr=learning_rate)

    #opt1 = torch.optim.SGD(model_data.parameters(), lr=0.0005, momentum=0.9)
    #opt2 = torch.optim.SGD(model_knowledge.parameters(), lr=0.0005, momentum=0.9)
    #opt3 = torch.optim.SGD(FC3.parameters(), lr=0.0005, momentum=0.9)

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
        for _, (a, b) in enumerate(zip(train_loader, feature_train_loader)):
            x, y = a
            feature_x, _ = b
            feature_x = feature_x.to(device)
            feature_x = model_knowledge(feature_x)
            x = x.to(device)
            y = y.to(device)
            deep_feature_x = model_data(x)
            #deep_feature_x = FC2(deep_feature_x)
            '''
            print('deep_feature_x')
            print(deep_feature_x)
            print('feature_x')
            print(feature_x)
            input('')
            '''
            #np.set_printoptions(threshold=sys.maxsize)
            #torch.set_printoptions(profile="full")
            #print('#' * 30)
            #print(deep_feature_x.shape)
            #print('#' * 30)
            #print(feature_x.shape)
            #input('')

            '''
            # Loss that penalizes more when deep_feature_x looks more different from feature_x
            loss_diff = torch.abs(torch.mean(torch.sqrt(deep_feature_x)) - torch.mean(torch.sqrt(feature_x)))
            print(torch.mean(torch.sqrt(deep_feature_x.detach().cpu())) , torch.mean(torch.sqrt(feature_x.detach().cpu())) , loss_diff)

            catted = torch.cat((deep_feature_x, feature_x), 1)
            #catted = Conv(catted)
            outputs = FC3(catted)
            #outputs = model_fusion((deep_feature_x, feature_x))
            loss = criterion(outputs, y)
            factor = 0.5
            loss += loss_diff * factor
            '''

            catted = torch.cat((deep_feature_x, feature_x), 1)
            #catted = Conv(catted)
            outputs = FC3(catted)
            softmax_out = torch.nn.Softmax(dim=1)(outputs / 2)  # TODO temperature with softmax
            loss = criterion(softmax_out, y)
            #loss = criterion(outputs, y)
            total_loss += loss
            cnt += 1

            # Backward and optimize
            #opt1.zero_grad()
            #opt2.zero_grad()
            #opt3.zero_grad()
            opt.zero_grad()
            loss.backward()
            #opt1.step()
            #opt2.step()
            #opt3.step()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

        if (epoch + 1) % 10 == 0:
            model_knowledge.eval()
            model_data.eval()
            FC3.eval()
            y_true = []
            y_pred = []
            test_cnt = 0
            with torch.no_grad():
                for i, (a, b) in enumerate(zip(test_loader, feature_test_loader)):
                    x, y = a
                    feature_x, _ = b
                    feature_x = feature_x.to(device)
                    feature_x = model_knowledge(feature_x)
                    x = x.to(device)
                    y = y.to(device)
                    deep_feature_x = model_data(x)
                    # deep_feature_x = FC2(deep_feature_x)
                    catted = torch.cat((deep_feature_x, feature_x), 1)
                    # catted = Conv(catted)
                    outputs = FC3(catted)
                    # outputs = model_fusion((deep_feature_x, feature_x))
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                    test_cnt += 1
                score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
                print('eval score:', score)
            model_knowledge.train()
            model_data.train()
            FC3.train()

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
            torch.save(model_data, './runs/' + str(dataset) + '/middlefusion_T_doubledropout_' + 'model_data' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            torch.save(model_knowledge,
                       './runs/' + str(dataset) + '/middlefusion_' + 'model_knowledge_T_doubledropout_' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            #torch.save(FC1, './runs/' + str(dataset) + '/middlefusion_' + 'FC1' + '_testsubjID_' + str(test_subj_id) +
            #           '_epoch_' + str(epoch + 1) + '.ckpt')
            #torch.save(FC2, './runs/' + str(dataset) + '/middlefusion_' + 'FC2' + '_testsubjID_' + str(test_subj_id) +
            #           '_epoch_' + str(epoch + 1) + '.ckpt')
            #torch.save(Conv1, './runs/' + str(dataset) + '/middlefusion_' + 'Conv1' + '_testsubjID_' + str(test_subj_id) +
            #           '_epoch_' + str(epoch + 1) + '.ckpt')
            torch.save(FC3, './runs/' + str(dataset) + '/middlefusion_' + 'FC_T_doubledropout_' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            #torch.save(model_fusion, './runs/' + str(dataset) + '/middlefusion_' + 'model_fusion' + '_testsubjID_' + str(test_subj_id) +
            #            '_epoch_' + str(epoch + 1) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break
    '''
    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_testsubjID_' + str(test_subj_id) + '_best.ckpt')
    '''
    #model.eval()
    #FC1.eval()
    #FC2.eval()
    #Conv.eval()
    model_knowledge.eval()
    model_data.eval()
    FC3.eval()
    #model_fusion.eval()
    y_true = []
    y_pred = []
    cnt = 0
    with torch.no_grad():
        for i, (a, b) in enumerate(zip(test_loader, feature_test_loader)):
            x, y = a
            feature_x, _ = b
            feature_x = feature_x.to(device)
            feature_x = model_knowledge(feature_x)
            x = x.to(device)
            y = y.to(device)
            deep_feature_x = model_data(x)
            #deep_feature_x = FC2(deep_feature_x)
            catted = torch.cat((deep_feature_x, feature_x), 1)
            #catted = Conv(catted)
            outputs = FC3(catted)
            #outputs = model_fusion((deep_feature_x, feature_x))
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
            cnt += 1
        score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))

        return score


def nn_fixepoch_SFN(
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
        class_out,
        ch_num=None,
        loss_weights=None
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

    batch_size = 32 # TODO

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

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
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)
    model.training = True
    #opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if label_probs:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(middle_feature_train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(middle_feature_train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet': # TODO
        feature_tensor_train_x = feature_tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_train_dataset = TensorDataset(feature_tensor_train_x, feature_tensor_train_y)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=batch_size)

    feature_tensor_test_x, feature_tensor_test_y = torch.from_numpy(middle_feature_test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet': # TODO
        feature_tensor_test_x = feature_tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_test_dataset = TensorDataset(feature_tensor_test_x, feature_tensor_test_y)
    feature_test_loader = DataLoader(feature_test_dataset, batch_size=batch_size)

    all_parameters = list(model.parameters())
    opt = torch.optim.Adam(all_parameters, lr=learning_rate)

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        total_loss_dis = 0
        cnt = 0
        for _, (a, b) in enumerate(zip(train_loader, feature_train_loader)):
            data, y = a
            knowledge, _ = b
            knowledge = knowledge.to(device)
            data = data.to(device)
            y = y.to(device)
            tup = (knowledge, data)
            outputs = model(tup)
            softmax_out = torch.nn.Softmax(dim=1)(outputs / 2)  # TODO temperature with softmax
            clf_loss = criterion(softmax_out, y)
            #loss = criterion(outputs, y)

            x1 = knowledge.cpu().numpy()
            x2 = data.cpu().numpy()

            np.save('./x1', x1)
            np.save('./x2', x2)

            sys.exit(0)

            k1, k = knowledge[0].cpu().numpy(), knowledge[1:].cpu().numpy()
            d1, d = data[0].cpu().numpy(), data[1:].cpu().numpy()


            dist1 = cosine_distances(k1.reshape(1, -1), k.reshape(k.shape[0], -1))
            dist2 = cosine_distances(d1.reshape(1, -1), d.reshape(d.shape[0], -1))
            dist_diff = np.abs(dist1 - dist2)
            #print(dist1)
            #print(dist2)
            #print(dist_diff)


            discrepancy_loss = np.sum(dist_diff) / (len(data) - 1)
            #print(discrepancy_loss)
            #input('')
            #print('discrepancy_loss:', discrepancy_loss)
            total_loss_dis += discrepancy_loss

            loss = clf_loss + discrepancy_loss * 50    # 10 , 50

            total_loss += loss
            cnt += 1

            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt
        out_loss_dis = total_loss_dis / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Discrepancy Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss, out_loss_dis))

        if (epoch + 1) % 10 == 0:
            model.eval()
            model.training = False
            y_true = []
            y_pred = []
            test_cnt = 0
            with torch.no_grad():
                for i, (a, b) in enumerate(zip(test_loader, feature_test_loader)):
                    data, y = a
                    knowledge, _ = b
                    knowledge = knowledge.to(device)
                    data = data.to(device)
                    y = y.to(device)
                    tup = (knowledge, data)
                    outputs = model(tup)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                    test_cnt += 1
                score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
                print('eval score:', score)
            model.train()
            model.training = True

        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/FN_feature_dual50' + str(batch_size) + '' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')

            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    model.eval()
    model.training = False
    y_true = []
    y_pred = []
    cnt = 0
    with torch.no_grad():
        for i, (a, b) in enumerate(zip(test_loader, feature_test_loader)):
            data, y = a
            knowledge, _ = b
            knowledge = knowledge.to(device)
            data = data.to(device)
            y = y.to(device)
            tup = (knowledge, data)
            outputs = model(tup)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
            cnt += 1
        score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]

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
