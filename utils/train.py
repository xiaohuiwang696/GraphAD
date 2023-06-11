import os
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import *
from utils.constants import color
from utils.parser import params, parser
from utils.pot import pot_eval
from utils.process import collate, GraphDataset
from utils.result import mergePOT, getTimePOT, getstampresult

datasetfolder = '../data/'


def load_dataset(dataset):
    """

    Args:
        dataset: 数据名字，读取processed数据，内容为train，test，label，

    Returns:

    """
    folder = os.path.join(datasetfolder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SWAT': file = args.swat_name + '_' + file
        if dataset == 'EV': file = args.ev_name + '_' + file
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'MIT': file = args.mit_name + '_' + file
        if dataset == 'WADI': file = 'wadi_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    return loader[0], loader[1], loader[2]


def GetDatasetName(dataset):
    if dataset == 'MIT':
        return args.mit_name
    elif dataset == 'EV':
        return args.ev_name
    elif dataset == 'SWAT':
        return args.swat_name
    else:
        return args.dataset


def save_model(model, optimizer, epoch, accuracy_list=None):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    datasetname = GetDatasetName(args.dataset)
    file_path = f'{folder}/model_{datasetname}_{args.slide_win}_{args.batch}.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(channel, time, tcnhidden):
    """

    Args:
        modelname: 模型名称
        dims: 数据的维度

    Returns: model, optimizer, scheduler, epoch, accuracy_list

    """
    import models.model
    modelname = args.model
    model_class = getattr(models.model, modelname)
    if args.model in ['TimeTransformer', 'TimeDeepInfo', 'CT_ED', 'Graph_ED']:
        model = model_class(channel, time)
    else:
        model = model_class(channel, time, [16, channel])

    # model = GraphAD(channel, time, tcnhidden)
    # model = TimeGAT(channel, time, tcnhidden)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    datasetname = GetDatasetName(args.dataset)
    fname = f'checkpoints/{args.model}_{args.dataset}/model_{datasetname}_{args.slide_win}_{args.batch}.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(
            f"{color.GREEN}Loading pre-trained model: model_{args.slide_win}_{args.batch} {color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, epoch, accuracy_list


def backprop(epoch, model, dataloader, optimizer, training=True):
    # device = torch.device("cpu")
    # print('\n device====>',torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    TimeMSE = nn.MSELoss(reduction='mean' if training else 'none').to(device)
    GraphMSE = nn.MSELoss(reduction='mean' if training else 'none').to(device)
    b_xent = nn.BCEWithLogitsLoss().to(device)
    l1s, l2s = [], []
    if model.name in ['GraphAD', 'GraphTransformer']:
        if training:
            for i, data in enumerate(dataloader):
                feats, batched_graph, matrix = data
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                matrix = matrix.to(device)
                if model.name == 'GraphAD':
                    time, adj = model(feats, batched_graph)
                else:
                    time, adj = model(feats)
                # loss = F.cross_entropy(logits, labels)
                Tloss = TimeMSE(time, feats)
                Gloss = GraphMSE(adj, matrix)
                loss = args.tratio * Tloss + args.gratio * Gloss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l1s.append(torch.mean(Tloss).item())
                l2s.append(torch.mean(Gloss).item())

            tqdm.write(f'Epoch {epoch},\t TimeLoss = {np.mean(l1s)},GraphLoss = {np.mean(l2s)}')
            return np.mean(l1s), np.mean(l2s)

        else:
            torch.zero_grad = False
            model.eval()
            l1s = []
            l2s = []
            for feats, batched_graph, matrix in dataloader:
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                matrix = matrix.to(device)
                if model.name == 'GraphAD':
                    time, adj = model(feats, batched_graph)
                else:
                    time, adj = model(feats)
                Tloss = TimeMSE(time, feats)
                Gloss = GraphMSE(adj, matrix)

                l1s.append(Tloss.detach().cpu().numpy())
                gmse = Gloss.detach().cpu().numpy()
                gmax = np.max(np.max(gmse, 1), 1)
                # print(gmax.shape)
                l2s.extend(gmax)
            l1s = np.vstack(l1s)
            l2s = np.array(l2s)
            l2s = l2s.flatten()
            return l1s, l2s
    elif model.name in ['TimeGAT', 'Graph_ED']:
        if training:
            for i, data in enumerate(dataloader):
                feats, batched_graph, matrix = data
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                time = model(feats, batched_graph)
                loss = TimeMSE(time, feats)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l1s.append(torch.mean(loss).item())
            tqdm.write(f'Epoch {epoch},\t TimeLoss = {np.mean(l1s)}')
            return np.mean(l1s), None

        else:
            torch.zero_grad = False
            model.eval()
            l1s = []
            for feats, batched_graph, matrix in dataloader:
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                time = model(feats, batched_graph)
                loss = TimeMSE(time, feats)
                l1s.append(loss.detach().cpu().numpy())
            l1s = np.vstack(l1s)
            return l1s, None
    elif model.name == 'GraphGAT':
        if training:
            for i, data in enumerate(dataloader):
                feats, batched_graph, matrix = data
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                matrix = matrix.to(device)
                graph = model(feats, batched_graph)
                loss = GraphMSE(graph, matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l1s.append(torch.mean(loss).item())
            tqdm.write(f'Epoch {epoch},\t GraphLoss = {np.mean(l1s)}')
            return np.mean(l1s), None

        else:
            torch.zero_grad = False
            model.eval()
            l1s = []
            for feats, batched_graph, matrix in dataloader:
                feats = feats.to(device)
                batched_graph = batched_graph.to(device)
                matrix = matrix.to(device)
                graph = model(feats, batched_graph)
                loss = GraphMSE(graph, matrix)
                gmse = loss.detach().cpu().numpy()
                gmax = np.max(np.max(gmse, 1), 1)
                # print(gmax.shape)
                l2s.extend(gmax)
            l2s = np.array(l2s)
            l2s = l2s.flatten()
            return None, l2s

    elif model.name in ['TimeTransformer', 'CT_ED']:
        if training:
            for i, data in enumerate(dataloader):
                feats = data
                feats = feats.to(device)
                time = model(feats)
                loss = TimeMSE(time, feats)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l1s.append(torch.mean(loss).item())
            tqdm.write(f'Epoch {epoch},\t TimeLoss = {np.mean(l1s)}')
            return np.mean(l1s), None

        else:
            torch.zero_grad = False
            model.eval()
            l1s = []
            for feats in dataloader:
                feats = feats.to(device)
                time = model(feats)
                loss = TimeMSE(time, feats)
                l1s.append(loss.detach().cpu().numpy())
            l1s = np.vstack(l1s)
            return l1s, None


def run():
    # print(params.pred, params.n_window, params.batch_size)
    trainD, testD, labels = load_dataset(args.dataset)
    # trainD=trainD[:100]
    length, channel = trainD.shape
    # trainD = trainD[:100]
    # testD = testD[:100]
    # validD = trainD[90:]
    # trainD = trainD[:90]

    trainL = int(length * 0.9)
    validD = trainD[trainL:]
    # trainD = trainD[:trainL]

    print(trainD.shape)
    print(validD.shape)
    print(testD.shape)

    nwindow = args.slide_win
    slide = args.slide_stride

    if args.model in ['TimeGAT', 'GraphAD', 'GraphTransformer', 'GraphGAT', 'Graph_ED']:
        trainDataset = GraphDataset(trainD, nwindow, slide)
        validDataset = GraphDataset(trainD, nwindow, slide)
        testDataset = GraphDataset(testD, nwindow, slide)
        train_dataloader = DataLoader(trainDataset, batch_size=args.batch, collate_fn=collate)
        valid_dataloader = DataLoader(validDataset, batch_size=args.batch, collate_fn=collate)
        test_dataloader = DataLoader(testDataset, batch_size=args.batch, collate_fn=collate)
    else:
        datatype = 'time'
        trainDataset = GraphDataset(trainD, nwindow, slide, datatype=datatype)
        # validDataset = GraphDataset(trainD, nwindow, slide, datatype='time')
        testDataset = GraphDataset(testD, nwindow, slide, datatype=datatype)
        train_dataloader = DataLoader(trainDataset, batch_size=args.batch)
        # valid_dataloader = DataLoader(validDataset, batch_size=args.batch)
        test_dataloader = DataLoader(testDataset, batch_size=args.batch)

    model, optimizer, epoch, accuracy_list = load_model(channel, nwindow, tcnhidden=[16, channel])
    ### Training phase
    # print(model.parameters)
    timeloss = 0
    graphloss = 0
    testflag = False
    # if not args.test:
    if testflag:
        # if train_flag:
        print(f'{color.HEADER}Training {model.name} on {args.dataset}{color.ENDC}')
        num_epochs = args.epoch;
        e = epoch + 1;
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, train_dataloader, optimizer, True)
            accuracy_list.append((lossT, lr))
            timeloss, graphloss = lossT, lr
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, e, accuracy_list)
        # valid
        # torch.zero_grad = True
        # model.eval()
        # print(f'{color.HEADER}Validing {args.model} on {args.dataset}{color.ENDC}')
        # tloss, gloss = backprop(0, model, valid_dataloader, optimizer, training=False)
        # print(f'Valid Loss: \t TimeLoss = {np.mean(tloss)}')

    ### Testing phase
    # print(f'{color.HEADER}Validing {model.name} on {args.dataset}{color.ENDC}')
    # tloss0, gloss0 = backprop(0, model, train_dataloader, optimizer, training=False)
    # print(f'Valid Loss: \t TimeLoss = {np.mean(tloss0)}')

    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {model.name} on {args.dataset}{color.ENDC}')
    tloss, gloss = backprop(0, model, test_dataloader, optimizer, training=False)
    tloss0, gloss0 = backprop(0, model, train_dataloader, optimizer, training=False)
    # print(tloss.shape)
    # print(gloss.shape)
    # loss.shape=(28479,38)

    ### Scores

    print(f'{color.HEADER}loading labels ……  {color.ENDC}  ')
    if args.dataset in ['SMD', 'EV']:
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    else:
        labelsFinal = labels
    labellength = testDataset.__len__()
    labelsFinal = labelsFinal[-labellength:]

    # mergePOT(gloss0, gloss, tloss0, tloss, labelsFinal)
    # getstampresult(tloss0, tloss, labelsFinal)
    if args.model in ['GraphAD', 'GraphTransformer']:
        result = mergePOT(gloss0, gloss, tloss0, tloss, labelsFinal, args.gratio)
        pd_reuslt = pd.DataFrame(result)
        pd_reuslt['model'] = args.model
        pd_reuslt['epoch'] = epoch
        pd_reuslt['batch'] = args.batch
        pd_reuslt['window'] = nwindow
        pd_reuslt['Gratio'] = args.gratio
        pd_reuslt['Tloss'] = timeloss
        pd_reuslt['Gloss'] = graphloss
        pd_reuslt['data'] = GetDatasetName(args.dataset)
        pd_reuslt.to_csv(f'result/{args.model}_results.csv', mode='a', index=None, header=None)
    elif args.model == 'GraphGAT':
        pd_reuslt = pd.DataFrame()
        gmaxresult, pred = pot_eval(gloss0, gloss, labelsFinal)
        pd_reuslt = pd_reuslt.append(gmaxresult, ignore_index=True)
        print(gmaxresult)
        pd_reuslt['model'] = args.model
        pd_reuslt['epoch'] = epoch
        pd_reuslt['batch'] = args.batch
        pd_reuslt['window'] = nwindow
        pd_reuslt['Gloss'] = graphloss
        pd_reuslt['data'] = GetDatasetName(args.dataset)
        pd_reuslt.to_csv(f'result/{args.model}_results.csv', mode='a', index=None, header=None)
    else:
        result = getTimePOT(tloss0, tloss, labelsFinal)
        pd_reuslt = pd.DataFrame(result)
        pd_reuslt['model'] = args.model
        pd_reuslt['epoch'] = epoch
        pd_reuslt['batch'] = args.batch
        pd_reuslt['window'] = nwindow
        pd_reuslt['batch'] = args.gratio
        pd_reuslt['data'] = GetDatasetName(args.dataset)
        pd_reuslt.to_csv(f'result/{args.model}_results.csv', mode='a', index=None, header=None)


args = parser.parse_args()
if __name__ == "__main__":
    print('device====>', torch.cuda.is_available())

    # random.seed(args.random_seed)
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # print('device====>', torch.cuda.is_available())
    args = parser.parse_args()
    # # args.model = 'CT_ED'
    # for i in range(1):
    #     run()

    # models = ['GraphTransformer', 'TimeTransformer', 'TimeGAT', 'GraphGAT']
    models = ['GraphAD']
    datasets = ['EV', 'MIT','EV']
    datanames = ['ev24n', 'MIT','ev57']
    ratio = [0.1, 0.5, 1, 0.4, 0.6]
    for i in models:
        # args = parser.parse_args()
        args.model = i
        for j in range(2,3):
            dataset = datasets[j]
            args.dataset = dataset
            if dataset == 'EV':
                args.slide_win = 12
                args.ev_name = datanames[j]
            elif dataset == 'MIT':
                args.slide_win = 6
                args.mit_name = datanames[j]
            else:
                args.slide_win = 10
                args.swat_name = datanames[j]
            for r in ratio:
                args.gratio = r
                for k in range(10):
                    run()

    # for i in range(15):
    #     args = parser.parse_args()
    #     args.model = 'CT_ED'
    #     run()
    # # for j in range(1, 3):
    # #     params.nwindow = 6 * j
    # #     # test()
    # #     for i in range(10):
    # #         run()
