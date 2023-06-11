import os
import random

import numpy as np
import torch


from utils.parser import parser, params
from utils.process import ConstructGraph
from utils.train import run

if __name__ == "__main__":

    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    print('device====>', torch.cuda.is_available())
    # args = parser.parse_args()
    # args.model = 'CT_ED'
    for i in range(1):
        run()
    # for j in range(1, 3):
    #     params.nwindow = 6 * j
    #     # test()
    #     for i in range(10):
    #         run()
