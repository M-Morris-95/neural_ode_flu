# import tensorflow as tf
import numpy as np
import metrics
from torchdiffeq import odeint_adjoint as odeint
import torch
import torch.nn as nn

import time

from parser import *
from data_loader import *
# from model import *

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dtype = 'torch.cuda.FloatTensor'
else:
    dev = torch.device("cpu")
    dtype = 'torch.FloatTensor'


parser = parser()
args = parser.parse_args()


class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.L1 = nn.Linear(dim, 32)
        self.activation = nn.Sigmoid()
        self.L2 = nn.Linear(32, dim)
        self.nfe = 0

    def forward(self, t, inputs):
        self.nfe += 1
        output = self.L1(inputs)
        output = self.activation(output)
        output = self.L2(output)
        return output

class ODEBlock(nn.Module):
    def __init__(self, odefunc, dim):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc(32)
        self.L1 = nn.GRUCell(dim, 32)
        self.activation = nn.ReLU()

    def forward(self, inputs):

        integration_time = torch.tensor(np.asarray([np.linspace(0, 27, 28), np.linspace(1, 28, 28)]).T).type(dtype)
        integration_time = torch.cat((integration_time, torch.tensor([[28,42]]).type(dtype)),0)
        hx = torch.randn(inputs.shape[0], 32)

        output = []

        hx = odeint(self.odefunc, hx, integration_time[0], rtol=args.tol, atol=args.tol)
        for i in range(inputs.shape[1]):
            hx = self.L1(inputs[:,i,:], hx[-1])
            hx = odeint(self.odefunc, hx, integration_time[i+1], rtol=args.tol, atol=args.tol)
            output.append(hx)

        output = self.activation(output[-1])
        return output

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

results = pd.DataFrame(index = ['CRPS','NLL','MAE','RMSE','SMAPE','Corr','MB Log','SDP'])
for fold_num in range(1,2):
    data = data_loader(args, dtyoe=dtype, fold=fold_num)

    train, test = data.build()

    trainloader = torch.utils.data.DataLoader(train,
                                              batch_size=32,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=32,
                                             shuffle=False)

    feature_layers = [ODEBlock(ODEfunc, 176)]
    fc_layers = [nn.BatchNorm1d(32),
                 nn.Linear(32, 1),
                 nn.ReLU(inplace=True),]

    model = nn.Sequential(*feature_layers, *fc_layers)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()

            running_loss = (i) / (i+1) * running_loss + loss.item() / (i+1)
            print('epoch: %d, batch: %d, loss: %.3f' % (epoch + 1, i+1, running_loss), end = '\r')
        print('')
        # print('epoch: %d, batch: %d, loss: %.3f' % (epoch + 1, i + 1, running_loss))


    # prediction = model(test[:][0])
    # print(torch.sqrt(criterion(prediction, test[:][1])))
    #
    #
    #
    # # my_nn = Net()
    # print(my_nn)


#     model = model_builder(x_train, y_train, args)
#     model.fit(x_train, y_train)
#     pred = model.predict(x_test, y_test)
#
#     results[str(fold_num + 2013) + '/' + str(fold_num + 14)] = [metrics.crps(pred),
#                                                                 metrics.nll(pred),
#                                                                 metrics.mae(pred),
#                                                                 metrics.rmse(pred),
#                                                                 metrics.smape(pred),
#                                                                 metrics.corr(pred),
#                                                                 metrics.mb_log(pred),
#                                                                 metrics.sdp(pred)]
#     tf.keras.backend.clear_session()
#
# results['Average'] = results.mean(1)
# results['Average'].loc['SDP'] = np.abs(results.loc['SDP'].values[-1]).mean()
#
# plt.plot(pred.index, pred['True'], color='black')
# plt.plot(pred.index, pred['Pred'], color = 'red')
# plt.fill_between(pred.index, pred['Pred']-pred['Std'], pred['Pred']+pred['Std'], color='pink', alpha=0.5)
# plt.show()
