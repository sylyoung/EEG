import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from utils import _classification as evaluate
from EEGNet import EEGNet


def train(data=None, cuda=False, learning_rate=0.001, num_iterations=100):
    tensor_train_x, tensor_train_y = data
    seed = 1
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        print('using cuda...')

    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=64)

    # TODO 512
    model = EEGNet(19, 512, 2)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_iterations):
        print('epoch:', epoch + 1)
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y.long())
            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_iterations, out_loss))
        if (epoch + 1) % 10 == 0 and epoch != 0:
            # Save the model checkpoint
            torch.save(model.state_dict(), './Checkpoint/eegnet_crosssubject_noEA' + str(epoch + 1) + '.ckpt')
    return model


def test(data=None, model=None, model_path=None, cuda=False):  # model or model_path
    if model is None:
        model = EEGNet(19, 512, 2)  # 定义模型结构
        model.load_state_dict(torch.load(model_path))  # 读取模型参数
    # Test the model
    model.eval()
    tensor_test_x, tensor_test_y = data
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)
    seed = 1
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        print('using cuda...')
    with torch.no_grad():
        correct = 0
        total = 0
        y_test = []
        y_pred = []
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)  # return one-hot predict classification
            _, pred = torch.max(outputs.data, 1)  # TODO: add AUC, Sen, ...
            total += y.size(0)
            print('y.size(0): ', y.size(0))
            correct += (pred == y).sum().item()
            # tensor to numpy
            y = y.numpy()
            pred = pred.numpy()
            y_test.append(y)
            y_pred.append(pred)
        # list to numpy
        y_t = np.array(y_test)
        y_p = np.array(y_pred)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        out = round(100 * correct / total, 3)
        acc = round(accuracy_score(y_t, y_p), 3)
        sen = round(recall_score(y_t, y_p, pos_label=1), 3)
        spec = round(evaluate.specificity_score(y_t, y_p), 3)
        try:
            auc = round(roc_auc_score(y_t, y_p), 3)
        except ValueError:
            pass
        f1 = round(f1_score(y_t, y_p, pos_label=1), 3)
        return out, acc, sen, spec, auc, f1
