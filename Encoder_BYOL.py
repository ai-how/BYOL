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
