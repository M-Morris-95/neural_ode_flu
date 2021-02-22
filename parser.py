import argparse

def parser():
    parser = argparse.ArgumentParser(
        description='Forecasting')
    parser.add_argument('--Epochs', type=int, help='Number of epochs', default=50, required=False)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--Batch_Size', type=int, help='Batch_Size', default=64, required=False)
    parser.add_argument('--Root',type=str,help='root of data directory', required=True)
    parser.add_argument('--Arch', type=str, help='Which model use? FF, LSTM', default='FF',required=False)
    parser.add_argument('--Ext', type=str, help='Which extension to use? -v, -m, -d, -c', default='-c', required=False)
    parser.add_argument('--Gamma', type=int, help='How far ahead should it forecast? 7, 14, 21, All?', default=14, required=False)

    return parser