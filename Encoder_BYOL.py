import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from kornia import augmentation as augs
from kornia import filters, color
from torch.utils.data import TensorDataset, DataLoader


from torch.autograd import Variable

import pandas as pd

class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.no_features = args.no_features
        self.hidden_size = 64
        self.num_layers = args.num_layers
        self.bidir = args.bidir
        if self.bidir:
            self.direction = 2
        else: self.direction = 1
        self.dropout = 0.0
        self.lstm = nn.LSTM(input_size=self.no_features, hidden_size=self.hidden_size, dropout=self.dropout, num_layers=self.num_layers, bidirectional=self.bidir)

    def forward(self, x, seq_len):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.cuda()
        
        ##sort the batch!
        seq_len, idx = seq_len.sort(0, descending=True)
        x = x[idx,:]    

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # undo the packing operation
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Decode the hidden state of the last time step     
        last_step_index_list = (seq_len - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1) #tensor size: (batch_size,1,hidden_size)
        #print (last_step_index_list)
        hidden_outputs = out.gather(1, last_step_index_list).squeeze() #tensor size: (batch_size,hidden_size)

        #unsort the batch!
        _, idx = idx.sort(0, descending=False)
        hidden_outputs = hidden_outputs[idx, :]

        return hidden_outputs
# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t #t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

# loss fn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
mse = nn.MSELoss()
cse = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

def loss_fn(x, y):
    loss = cos(x,y)
    mse_loss = mse(x,y)
    lbl = torch.ones(y.shape[0])
    cse_loss = cse(x,y,lbl)
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return  cse_loss #-2 * (x * y).sum(dim=-1) mse_loss

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    print ("updating model weights")
    print (ma_model.net.lstm.weight_ih_l0[:5])
    print (current_model.net.lstm.weight_ih_l0[:5])
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
    print (ma_model.net.lstm.weight_ih_l0[:5])

# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self._register_hook()

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x, seq_len):
        if self.layer == -1:
            return self.net(x, seq_len)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, seq_len):
        representation = self.get_representation(x, seq_len)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection

# main class

class BYOL(nn.Module):
    def __init__(self, net, image_size, hidden_layer = -1, projection_size = 256, projection_hidden_size = 4096, augment_fn = None, moving_average_decay = 0.99):
        super().__init__()
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.hidden_layer = hidden_layer

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        
        self.target_encoder = None #NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer) #None
        
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(64, 96), seq_len=sequences)
        
    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        target_encoder._register_hook()
        return target_encoder

    '''@singleton('target_encoder')
    def _get_target_encoder(self):
            
        state = self.online_encoder.state_dict()
        state_clone = copy.deepcopy(state)

        target_encoder = self.target_encoder.load_state_dict(state_clone, strict=True)
        
        #target_encoder._register_hook()
        return target_encoder'''

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
        
    def mask_ts(self, x, p=.2):
        mask = torch.bernoulli((1.0 - p) * torch.ones(x.shape))
        return x.mul(mask)

    def forward(self, x, seq_len):
        
        image_two = image_one = self.mask_ts(x, p=.2) ## keeping augmented view same for testing code
        
        image_one = Variable(image_one.unsqueeze(dim=-1).type(FloatTensor))
        
        #image_two = self.mask_ts(x, p=.8)
        
        image_two = Variable(image_two.unsqueeze(dim=-1).type(FloatTensor))

        online_proj_one = self.online_encoder(image_one, seq_len)
        online_proj_two = self.online_encoder(image_two, seq_len)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        
        with torch.no_grad():
            
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one, seq_len)
            target_proj_two = target_encoder(image_two, seq_len)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss, loss_one, loss_two, online_proj_two, target_proj_two

cuda =torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

print("Begin training.")

#train_iter = iter(train_loader)

class all_arguments():
    no_features = 1
    hidden_size = 64
    num_layers = 1
    bidir = False
    batch_size = 64
    lr = 1e-3
    eps = 1e-3
    margin = 1
    batch_size = 10
    
args=all_arguments()
batch_size = 64
net = LSTMEncoder(args)

train = pd.read_csv('ElectricDevices_TRAIN.csv', header=None)
train[0] = train.apply(lambda row: row[0]-1, axis=1)
train['len'] = 96

train_loader = DataLoader(train.values,batch_size,shuffle=True)

out = next(iter(train_loader))
x_tr, x_tr_seq, y_tr = out[:, 1:97], torch.reshape(out[:,97].long(), (-1,)), torch.reshape(out[:,0].long(), (-1,))
sequences = Variable(x_tr_seq.long())

model = BYOL(
        net,
        image_size = 52,
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.5
    )

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for e in range(500, 500+20):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        #next(iter(train_loader))
        out = next(iter(train_loader))
        x_tr, x_tr_seq, y_tr = out[:, 1:97], torch.reshape(out[:,97].long(), (-1,)), torch.reshape(out[:,0].long(), (-1,))

        optimizer.zero_grad()

        #inp = Variable(x_tr.unsqueeze(dim=-1).type(FloatTensor)) 
        sequences = Variable(x_tr_seq.long())

        train_loss, loss_1, loss_2, img1, img2 = model(x_tr,sequences)
        #if e%1==0:
        #print (img1)
        #print (img2)
        #print (train_loss.sum(), loss_1, loss_2, img1, img2)
        train_loss.sum().backward()
        #print (model.target_encoder.net.lstm.weight_hh_l0.detach().numpy())
        #print (model.online_encoder.net.lstm.weight_hh_l0.detach().numpy())
        optimizer.step()
        print ("==========e==========")
        print ("---online---", model.online_encoder.net.lstm.weight_ih_l0[:2].tolist())
        print ("---target---",model.target_encoder.net.lstm.weight_ih_l0[:2].tolist())
        #model.update_moving_average()
        #print ("loss: ", y_train_pred)
