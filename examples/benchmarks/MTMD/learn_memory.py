import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
# regiodatetimeG_CN, REG_US]
from qlib.config import REG_US, REG_CN
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gats import GATModel
from qlib.contrib.model.pytorch_sfm import SFM_Model
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.model.pytorch_transformer import Transformer
from model2 import *
from utils import metric_fn, mse
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    if model_name.upper() == 'GATS':
        return GATModel

    if model_name.upper() == 'SFM':
        return SFM_Model

    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'HIST':
        return HIST

    raise ValueError('unknown model name `%s`'%model_name)

def gather_loss(query, keys):

    batch_size,dims = query.size() # b X h X w X d

    loss_mse = torch.nn.MSELoss()

    softmax_score_query, softmax_score_memory = get_score(keys, query)
 
    query_reshape = query.contiguous().view(batch_size, dims)

    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

    gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

    return gathering_loss

def spread_loss(query, keys):
    batch_size, dims = query.size() # b X h X w X d

    loss = torch.nn.TripletMarginLoss(margin=1.0)

    softmax_score_query, softmax_score_memory = get_score(keys, query)

    query_reshape = query.contiguous().view(batch_size, dims)

    _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

    #1st, 2nd closest memories
    pos = keys[gathering_indices[:,0]]
    neg = keys[gathering_indices[:,1]]

    spreading_loss = loss(query_reshape,pos.detach(), neg.detach())

    return spreading_loss

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])


global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix = None,m_items = None):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        #
        global_step += 1
        feature, label, market_value , stock_index, _ = train_loader.get(slc)
        if args.model_name == 'HIST':
            pred,m_items = model(feature, stock2concept_matrix[stock_index], market_value,m_items,train=True)
        else:
            pred = model(feature)
        
        loss = loss_fn(pred, label, args)
        
        #loss += (loss_gather_loss*0.1)#+)(loss_spread_loss *0.1)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()
    return m_items

def test_epoch(rep,epoch, model, test_loader, writer, args, stock2concept_matrix=None,m_items = None, prefix='Test', train=False):

    model.eval()
    
    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'HIST':
                 pred, m_items = model(feature, stock2concept_matrix[stock_index], market_value,m_items,train=False)
            else:
                pred, m_items= model(feature,m_items,train=False)
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    if not os.path.exists(args.outdir+"/csv"):
        os.makedirs(args.outdir+"/csv")
    preds.to_csv(args.outdir+"/csv/+"+args.model_name+"_r"+str(rep)+"_e"+str(epoch)+'.csv')
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(model, data_loader, stock2concept_matrix=None,m_items = None, train=False):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred,m_items_out = model(feature, stock2concept_matrix[stock_index], market_value,m_items,False)
            else:
                pred = model(feature)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args):

    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}
    segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder,segments)

    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    import pickle5 as pickle
    with open(args.market_value_path, "rb") as fh:
        df_market_value = pickle.load(fh)
    #df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'], pin_memory=True, start_index=start_index, device = device)
    
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'], pin_memory=True, start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    stock2concept_matrix = np.load(args.stock2concept_matrix) 
    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(args.repeat):
       
        pprint('create model...')
        ######################### modification #########################
        m_item0 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_item1 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        # m_item2 = F.normalize(torch.rand((96,64), dtype=torch.float), dim=1).cuda()
        m_items = [m_item0, m_item1]
        #m_items = [m_item1]
        ######################### modification #########################
        if args.model_name == 'SFM':
            model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
        elif args.model_name == 'ALSTM':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
        elif args.model_name == 'Transformer':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
        elif args.model_name == 'HIST':
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
        else:
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        best_test_score = -np.inf
        best_test_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)
        #m_items=torch.load("m_items.bin.r1.e44")
        for epoch in range(args.n_epochs):
            pprint('Running', times,'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix,m_items)


            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)

            pprint('evaluating...')
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(times,epoch, model, train_loader, writer, args, stock2concept_matrix, m_items = m_items,prefix='Train', train=False)
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(times,epoch, model, valid_loader, writer, args, stock2concept_matrix,m_items = m_items, prefix='Valid', train=False)
            torch.save(model, output_path + '/model.bin'+'.r'+str(times)+'.e' + str(epoch))
            torch.save(optimizer, output_path + '/optimizer.bin'+'.r'+str(times)+'.e' + str(epoch))
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(times,epoch, model, test_loader, writer, args, stock2concept_matrix,m_items = m_items, prefix='Test', train=False)

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
            pprint('Train Precision: ', train_precision)
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            pprint('Train Recall: ', train_recall)
            pprint('Valid Recall: ', val_recall)
            pprint('Test Recall: ', test_recall)
            model.load_state_dict(params_ckpt)
            if test_score>best_test_score:
                best_test_score=test_score
                best_test_epoch=epoch
            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break
        pprint('best test score:', best_test_score, '@', best_test_epoch)
        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin')

class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='HIST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=2)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='csi300_HIST')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')

    parser.add_argument('--outdir', default='./output/csi300_HIST')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
