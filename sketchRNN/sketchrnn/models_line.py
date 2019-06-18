import torch
from torch import nn

from sketchrnn.modelsutils import get_distr_param


class EncoderRNN_line(nn.Module):
    def __init__(self, hp):
        super(EncoderRNN_line, self).__init__()
        self.hp = hp
        # bidirectional lstm:
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(hp.dropout)
        # create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            if self.hp.use_cuda:
                hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, self.hp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, self.hp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(self.dropout(hidden), 1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        # N ~ N(0,1)
        z_size = mu.size()
        if self.hp.use_cuda:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        z = mu + sigma*N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat


class DecoderRNN_line(nn.Module):
    def __init__(self, hp, max_len_out=10):
        # import pdb; pdb.set_trace()
        # TODO: error before the init of decoder
        super(DecoderRNN_line, self).__init__()
        self.hp = hp
        self.max_len_out = max_len_out
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size)
        self.dropout = nn.Dropout(hp.dropout)

        # size of the output
        self.output_size = 0
        # TODO: il me semble que ce devrait être des deux au lieu des trois, car il y a p_r,mu_r,phi_r.
        # @Louis: je modifie du me dira si tu es daccord
        self.output_size += 6 * hp.M  # 2D gaussian mixture parameters for the center of the line
        self.output_size += 3 * hp.Mr  # 1D gaussian mixture for the length of the line
        self.output_size += 3 * hp.Mphi  # 1D gaussian mixture for the angle of the line
        self.output_size += 1  # probability to end the image

        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, self.output_size)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in 'outputs', while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
            len_out = self.max_len_out + 1
        else:
            y = self.fc_params(self.dropout(hidden).view(-1, self.hp.dec_hidden_size))
            len_out = 1
        # check partially dimensions of y
        # TODO: I remove it but who knows?
        # if y.numel() != (self.max_len_out + 1) * self.output_size * self.hp.batch_size:
        #     raise ValueError('wrong number of elements in y')
        (coef_center, coef_rad, coef_ang, q) = get_distr_param(y, len_out, self.hp, type_param='line')
        return (*coef_center, *coef_rad, *coef_ang, q, hidden, cell)
