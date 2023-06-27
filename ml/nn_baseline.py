import math
import os
import random
import time

import numpy as np
import torch
import sklearn
from torch.utils.data import TensorDataset, DataLoader

from EEG.models.FC import FC
from EEG.utils.alg_utils import soft_cross_entropy_loss
from EEG.utils.data_utils import split_data


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
        loss_weights=None,
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

    batch_size = 32 # 32 128 256 # TODO

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
    if model_name == 'EEGNet' or model_name == 'BEEGNet':
        tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    if model_name == 'CE_stSENet':
        tensor_train_x = tensor_train_x.unsqueeze_(2)
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    if valid_percentage != None and valid_percentage > 0:
        if label_probs:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y).to(torch.float32)
        else:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y.reshape(-1,)).to(torch.long)
        if model_name == 'EEGNet' or model_name == 'BEEGNet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(3).permute(0, 3, 1, 2)
        if model_name == 'CE_stSENet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(2)
        valid_dataset = TensorDataset(tensor_valid_x, tensor_valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet' or model_name == 'BEEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    if model_name == 'CE_stSENet':
        tensor_test_x = tensor_test_x.unsqueeze_(2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if cuda:
        model.cuda()
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
            if cuda:
                x = x.cuda()
                y = y.cuda()

            #outputs, kl = model(x)  # BNN
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
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.cuda()
                    y = y.cuda()
                    #outputs, kl = model(x)  # BNN
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                '''
                print('test acc:', np.round(sklearn.metrics.accuracy_score(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(
                    -1, ).tolist(), 5)[0])
                print('test bca:', np.round(sklearn.metrics.balanced_accuracy_score(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(
                    -1, ).tolist(), 5)[0])
                '''
            model.train()

        if valid_percentage != None and valid_percentage > 0 and (epoch + 1) % 5 == 0:
            model.eval()
            total_loss_eval = 0
            cnt_eval = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    if cuda:
                        x = x.cuda()
                        y = y.cuda()
                    outputs, kl = model(x)  # BNN
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

        if (epoch + 1) % num_iterations == 0:
            print('no save')
            #torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) +
            #                      '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            #torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_EA_' + str(test_subj_id) +
            #           '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            #print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    '''
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tta = CoTTA(model, opt)

    y_true = []
    y_pred = []
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        outputs = tta(x)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(y.cpu())
        y_pred.append(predicted.cpu())
    score = \
    np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(),
             5)[0]
    print('score:', score)
    # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
    return score
    '''

    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        all_output = []
        for x, y in test_loader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            #outputs, kl = model(x)  # BNN
            outputs = model(x)
            all_output.append(outputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
        if metrics is None:
            all_output = torch.nn.Softmax(dim=1)(torch.cat(all_output))
            all_output = all_output.detach().cpu()
            predict = np.zeros(all_output.shape[0])
            for m in range(all_output.shape[0]):
                if all_output[m][1] >= 0.3:
                    predict[m] = 1.0
            all_label = np.concatenate(y_true).reshape(-1, )
            f1 = sklearn.metrics.f1_score(all_label, predict, average='weighted')  # 这个weighted一定要加
            bca = sklearn.metrics.balanced_accuracy_score(all_label, predict)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(all_label, predict).ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            sen = tp / (tp + fn)
            spec = tn / (tn + fp)
            all_label = np.eye(2)[all_label]
            try:
                auc = sklearn.metrics.roc_auc_score(all_label, all_output, average='weighted')
            except ValueError:
                pass
            return acc * 100, sen * 100, spec * 100, auc * 100, f1 * 100, bca * 100
        else:
            score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


def nn_fixepoch_ms(
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
        train_x,
        train_y,
        test_x,
        test_y,
        loss_weights=None,
        num_sources=None,
        deep_feature_size=None
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

    class_out = len(np.unique(train_y))

    sources_x = split_data(train_x, times=num_sources, axis=0)
    sources_y = split_data(train_y, times=num_sources, axis=0)
    train_loader = []
    clfs = []
    opt_clfs = []
    for i in range(num_sources):
        train_x = sources_x[i]
        train_y = sources_y[i]
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
        if model_name == 'EEGNet':
            tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
        source_dataset = TensorDataset(tensor_train_x, tensor_train_y)
        source_loader = DataLoader(source_dataset, batch_size=batch_size)
        train_loader.append(source_loader)

        # initialize a classifier (FC layer) for each source domain
        source_FC = FC(nn_in=deep_feature_size, nn_out=class_out)
        source_FC.cuda()
        clfs.append(source_FC)
        opt_clf = torch.optim.Adam(source_FC.parameters(), lr=learning_rate)
        opt_clfs.append(opt_clf)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0

        source_iters = [iter(loader) for loader in train_loader]
        for batch_id in range(len(source_iters[0])):
            for source_id in range(num_sources):
                x, y = next(source_iters[source_id])

                # Forward pass
                x = x.cuda()
                y = y.cuda()

                outputs = model(x)

                outputs = clfs[source_id](outputs)

                loss = criterion(outputs, y)

                total_loss += loss
                cnt += 1

                # Backward and optimize
                opt.zero_grad()
                opt_clfs[source_id].zero_grad()
                loss.backward()
                opt.step()
                opt_clfs[source_id].step()
        out_loss = total_loss / cnt

        #if (epoch + 1) % 10 == 0:
        #    print('Epoch [{}/{}], Loss: {:.4f}'
        #          .format(epoch + 1, num_iterations, out_loss))
        """
        if (epoch + 1) % 10 == 0:
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.cuda()
                    y = y.cuda()
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                print('test score:', np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(
                    -1, ).tolist(), 5)[0])
            model.train()
        """

        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/' + 'MultiSource_' + model_name + '_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))

    # Test the model
    model.eval()
    y_true = []
    y_pred = []
    softmax = torch.nn.Softmax(dim=1)
    #perf = [83.33,52.08,97.92,75.00,56.25,67.36,72.22,88.19,71.53,73.77]
    with torch.no_grad():
        for x, y in test_loader:
            all_probs = None
            votes = None
            #print('#' * 10 + 'true:' + '#' * 10)
            #print(y)
            #print('#' * 25)
            for i in range(num_sources):
                x = x.cuda()
                y = y.cuda()
                outputs = model(x)
                outputs = clfs[i](outputs)
                predicted_probs = softmax(outputs)
                if all_probs is None:
                    all_probs = torch.zeros((x.shape[0], class_out)).cuda()
                else:
                    all_probs += predicted_probs.reshape(x.shape[0], class_out)
                #print('#' * 10 + 'source_number_' + str(i) + '#' * 10)
                #print(torch.max(predicted_probs.reshape(x.shape[0], class_out), 1)[1])
                #s = sklearn.metrics.accuracy_score(y.cpu(), torch.max(predicted_probs.reshape(x.shape[0], class_out), 1)[1].cpu())
                #print('ACC for this source classifier:', str(np.round(s * 100, 2)))
                #print('ACC on this source (data quality):', str(perf[i+1]))

                _, predicted = torch.max(predicted_probs, 1)

                if votes is None:
                    votes = torch.zeros((x.shape[0], class_out)).cuda()

                for i in range(x.shape[0]):
                    votes[i, predicted[i]] += 1

            #input('')
            #_, predicted = torch.max(all_probs, 1)
            _, predicted = torch.max(votes, 1)

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

    model_knowledge.cuda()
    model_data.cuda()
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

    #FC1 = FC(nn_in=middle_feature_train_x.shape[1], nn_out=middle_feature_train_x.shape[1]).cuda()
    #FC2 = FC(nn_in=feature_deep_dim, nn_out=feature_deep_dim).cuda()
    #Conv = Conv1().cuda()
    #Conv = ConvChannel(in_channels=tensor_feature_train_x.shape[1], out_channels=out_channels, kernel_size=1, bias=False, groups=1).cuda()
    FC3 = FC(nn_in=feature_deep_dim, nn_out=class_out).cuda()
    #FC3 = FC_2layer(nn_in=(middle_feature_train_x.shape[1] + feature_deep_dim), nn_out=class_out).cuda()
    #FC3 = FC(nn_in=(74 * out_channels + feature_deep_dim), nn_out=class_out).cuda()

    #model_fusion = ConvFusion(nn_deep=(74 * out_channels + feature_deep_dim), nn_out=class_out,
    #                in_channels=tensor_feature_train_x.shape[1], out_channels=out_channels,
    #                kernel_size=1, bias=False, groups=1).cuda()


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
            feature_x = feature_x.cuda()
            feature_x = model_knowledge(feature_x)
            x = x.cuda()
            y = y.cuda()
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
                    feature_x = feature_x.cuda()
                    feature_x = model_knowledge(feature_x)
                    x = x.cuda()
                    y = y.cuda()
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
                    x = x.cuda()
                    y = y.cuda()
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
            feature_x = feature_x.cuda()
            feature_x = model_knowledge(feature_x)
            x = x.cuda()
            y = y.cuda()
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

    model.cuda()
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
    #if model_name == 'EEGNet': # TODO
    #    feature_tensor_train_x = feature_tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_train_dataset = TensorDataset(feature_tensor_train_x, feature_tensor_train_y)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=batch_size)

    feature_tensor_test_x, feature_tensor_test_y = torch.from_numpy(middle_feature_test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    #if model_name == 'EEGNet': # TODO
    #    feature_tensor_test_x = feature_tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
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
        #total_loss_dis = 0
        cnt = 0
        for _, (a, b) in enumerate(zip(train_loader, feature_train_loader)):
            data, y = a
            knowledge, _ = b
            knowledge = knowledge.cuda()
            data = data.cuda()
            y = y.cuda()
            tup = (knowledge, data)
            #outputs = model(tup)
            (k_rep, d_rep), outputs = model(tup)
            temperature = 2
            softmax_out = torch.nn.Softmax(dim=1)(outputs / temperature)  # TODO temperature with softmax
            loss = criterion(softmax_out, y)
            #clf_loss = criterion(softmax_out, y)
            #loss = criterion(outputs, y)

            """
            k1, k = k_rep[0].detach().cpu().numpy(), k_rep[1:].detach().cpu().numpy()
            d1, d = d_rep[0].detach().cpu().numpy(), d_rep[1:].detach().cpu().numpy()
            dist1 = cosine_distances(k1.reshape(1, -1), k.reshape(k.shape[0], -1))
            dist2 = cosine_distances(d1.reshape(1, -1), d.reshape(d.shape[0], -1))
            dist_diff = np.abs(dist1 - dist2)
            discrepancy_loss = np.sum(dist_diff) / (len(data) - 1)
            total_loss_dis += discrepancy_loss
            loss = clf_loss + discrepancy_loss * 3   # 10 , 50
            """

            '''
            variance, sample_mean = torch.var_mean(k_rep)
            sub_map = torch.sub(k_rep, sample_mean)
            correlation_matrix_k = torch.div(sub_map, variance)

            variance, sample_mean = torch.var_mean(d_rep)
            sub_map = torch.sub(d_rep, sample_mean)
            correlation_matrix_d = torch.div(sub_map, variance)

            # k_rep_map_T = torch.transpose(correlation_matrix_k, 1, 0)
            # d_rep_map_T = torch.transpose(correlation_matrix_d, 1, 0)

            # rgb_sq_ft_map = k_rep_map_T.squeeze()
            # rgb_avg_sq_ft_map = torch.mean(rgb_sq_ft_map, 0)
            # depth_sq_ft_map = d_rep_map_T.squeeze()
            # depth_avg_sq_ft_map = torch.mean(depth_sq_ft_map, 0)

            k_corr = torch.mul(k_rep, k_rep)
            d_corr = torch.mul(d_rep, d_rep)

            corr_diff = torch.sqrt(torch.sum(torch.sub(k_corr, d_corr) ** 2))
            total_loss_dis += corr_diff

            loss = clf_loss + corr_diff * 0.1
            '''

            total_loss += loss
            cnt += 1

            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt
        #out_loss_dis = total_loss_dis / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

        #if (epoch + 1) % 10 == 0:
        #    print('Epoch [{}/{}], Loss: {:.4f}, Discrepancy Loss: {:.4f}'
        #          .format(epoch + 1, num_iterations, out_loss, out_loss_dis))

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
                    knowledge = knowledge.cuda()
                    data = data.cuda()
                    y = y.cuda()
                    tup = (knowledge, data)
                    (_, _), outputs = model(tup)
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.append(y.cpu())
                    y_pred.append(predicted.cpu())
                    test_cnt += 1
                score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
                print('eval score:', score)
            model.train()
            model.training = True

        if (epoch + 1) % 100 == 0:
            torch.save(model, './runs/' + str(dataset) + '/FN_dual3' + str(batch_size) + '' + '_testsubjID_' + str(test_subj_id) +
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
            knowledge = knowledge.cuda()
            data = data.cuda()
            y = y.cuda()
            tup = (knowledge, data)
            (_, _), outputs = model(tup)
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

    model.cuda()
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
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()

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
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()

            x = [x1, x2]
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.item())
            y_pred.append(predicted.item())
        score = np.round(metrics(y_true, y_pred), 5)
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


def regularizer(loss1, loss2):
    beta = 2.0
    if loss1 - loss2 > 0:
        return (beta * math.exp(loss1 - loss2)) - 1
    return 0.0

def cal_corr(feature_map):
    variance, sample_mean = torch.var_mean(feature_map)
    sub_map = torch.sub(feature_map, sample_mean)
    correlation_matrix = torch.div(sub_map, variance)
    return correlation_matrix

def cal_score(corr_1, corr_2, feature_map_1, feature_map_2, loss_1, loss_2):
    _lambda = 0.05

    feature_map_T_1 = torch.transpose(corr_1, 0, 1)
    feature_map_T_2 = torch.transpose(corr_2, 0, 1)

    rgb_corr = torch.mul(feature_map_1, feature_map_T_1)
    depth_corr = torch.mul(feature_map_2, feature_map_T_2)

    focal_reg_param_1 = regularizer(loss_1, loss_2)
    focal_reg_param_2 = regularizer(loss_2, loss_1)

    corr_diff_1 = torch.sqrt(torch.sum(torch.sub(rgb_corr, depth_corr) ** 2))
    corr_diff_2 = torch.sqrt(torch.sum(torch.sub(depth_corr, rgb_corr) ** 2))

    # loss (m,n)
    ssa_loss_1 = focal_reg_param_1 * corr_diff_1
    ssa_loss_2 = focal_reg_param_2 * corr_diff_2

    # total loss
    reg_loss_1 = loss_1 + (_lambda * ssa_loss_1)
    reg_loss_2 = loss_2 + (_lambda * ssa_loss_2)

    return reg_loss_1, reg_loss_2


def focal_reg_param(loss_from, loss_to, beta):
    diff = loss_from - loss_to
    return torch.exp(beta * diff-1) if diff > 0 else torch.tensor([.0])

class ssa_loss(torch.nn.Module):
    def __init__(self):
        super(ssa_loss, self).__init__()

    def forward(self, m, n, rho=1):
        m, n = torch.mean(m, dim=0), torch.mean(n, dim=0)
        m, n = m.view(-1, m.shape[0]), n.view(-1, n.shape[0])
        m_corr, n_corr = torch.matmul(m, m.t()), torch.matmul(n, n.t())
        total_loss = rho * torch.pow(torch.norm(m_corr - n_corr), 2)
        return total_loss

def nn_cotrain(
        model_k,
        model_d,
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
        feature_train_x,
        feature_test_x,
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

    model_k.cuda()
    model_d.cuda()
    if label_probs:
        criterion = soft_cross_entropy_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if loss_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if label_probs:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(feature_train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        feature_tensor_train_x, feature_tensor_train_y = torch.from_numpy(feature_train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        feature_tensor_train_x = feature_tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_train_dataset = TensorDataset(feature_tensor_train_x, feature_tensor_train_y)
    feature_train_loader = DataLoader(feature_train_dataset, batch_size=batch_size)

    feature_tensor_test_x, feature_tensor_test_y = torch.from_numpy(feature_test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        feature_tensor_test_x = feature_tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    feature_test_dataset = TensorDataset(feature_tensor_test_x, feature_tensor_test_y)
    feature_test_loader = DataLoader(feature_test_dataset, batch_size=batch_size)

    opt_k = torch.optim.Adam(list(model_k.parameters()), lr=learning_rate)
    opt_d = torch.optim.Adam(list(model_d.parameters()), lr=learning_rate)

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # early stopping
    valid_cnter = 0
    best_valid_loss = 2 ** 20
    stop = False

    # Train the model
    for epoch in range(num_iterations):
        total_loss_k = 0
        total_loss_d = 0
        cnt = 0
        for _, (a, b) in enumerate(zip(train_loader, feature_train_loader)):
            x, y = a
            feature_x, _ = b
            feature_x = feature_x.cuda()
            x = x.cuda()
            y = y.cuda()
            feature_map_k, outputs_k = model_k(feature_x)
            feature_map_d, outputs_d = model_d(x)

            #softmax_out = torch.nn.Softmax(dim=1)(outputs_k / 2)  # TODO temperature with softmax
            loss_k = criterion(outputs_k, y)
            loss_d = criterion(outputs_d, y)

            #corr_k = cal_corr(feature_map_k)
            #corr_d = cal_corr(feature_map_d)
            #reg_loss_k, reg_loss_d = cal_score(corr_k, corr_d, feature_map_k, feature_map_d, loss_k, loss_d)

            reg_k = focal_reg_param(loss_k, loss_d, beta=2.0)
            reg_d = focal_reg_param(loss_d, loss_k, beta=2.0)
            ssa1 = ssa_loss()
            ssa2 = ssa_loss()
            ssa_loss_k = ssa1(feature_map_k, feature_map_d)
            ssa_loss_d = ssa2(feature_map_d, feature_map_k)
            print(loss_k, ssa_loss_k * reg_k.item(), ssa_loss_k, reg_k.item())
            print(loss_d, ssa_loss_d * reg_d.item(), ssa_loss_d, reg_d.item())
            reg_loss_k = loss_k + ssa_loss_k * reg_k.item()
            reg_loss_d = loss_d + ssa_loss_d * reg_d.item()

            total_loss_k += reg_loss_k
            total_loss_d += reg_loss_d
            cnt += 1

            opt_k.zero_grad()
            opt_d.zero_grad()
            loss_k.backward()
            loss_d.backward()
            opt_k.step()
            opt_d.step()
        out_loss_k = total_loss_k / cnt
        out_loss_d = total_loss_d / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss_k: {:.4f}, Loss_d: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss_k, out_loss_d))

        if (epoch + 1) % 100 == 0:
            torch.save(model_k, './runs/' + str(dataset) + '/cotrain_' + 'model_data' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            torch.save(model_d,
                       './runs/' + str(dataset) + '/cotrain_' + 'model_knowledge' + '_testsubjID_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')

            print("--- %s seconds ---" % (time.time() - start_time))
        if stop:
            break

    #model_k.eval()
    model_d.eval()
    y_true = []
    y_pred = []
    cnt = 0
    with torch.no_grad():
        for x, y in test_loader:
            #x, y = a
            #feature_x, _ = b
            #feature_x = feature_x.cuda()
            x = x.cuda()
            y = y.cuda()
            #_, outputs_k = model_k(feature_x)
            _, outputs_d = model_d(x)
            _, predicted = torch.max(outputs_d.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
            cnt += 1
        score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]

        return score