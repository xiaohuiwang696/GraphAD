import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='MIT',
                    help="dataset from ['EV', 'SWaT']")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='GraphAD',
                    help="model name")

parser.add_argument('--test', action='store_true', help="test the model")
parser.add_argument('--retrain', action='store_true', help="retrain the model")
parser.add_argument('--less', action='store_true', help="train using less data")
parser.add_argument('--ev_name', help='data name', type=str, default='ev57')
parser.add_argument('--swat_name', help='data name', type=str, default='swat2')
parser.add_argument('--mit_name', help='data name', type=str, default='MIT')
parser.add_argument('-batch', help='batch size', type=int, default=256)
parser.add_argument('-epoch', help='train epoch', type=int, default=1)
parser.add_argument('-slide_win', help='slide_win', type=int, default=6)
parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
parser.add_argument('-topk', help='topk num', type=int, default=3)
parser.add_argument('-random_seed', help='random seed', type=int, default=0)
parser.add_argument('-tratio', help='Time Loss ratio', type=int, default=1)
parser.add_argument('-gratio', help='Graph Loss Ratio', type=int, default=0.1)
parser.add_argument('-disratio', help='Discriminator Loss Ratio', type=int, default=1)
parser.add_argument('-learning_rate', help='learning_rate', type=float, default=0.001)

args = parser.parse_args()


class params:
    nwindow = 10
