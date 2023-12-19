import torch.nn as nn
import torch
from custom.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.feature_size = feature_size
        
    def forward(self, src, tgt):
        tgt_mask = tgt.shape[1] if len(tgt.shape) == 3 else tgt.shape[0]
        
        decoder_mask = self._generate_square_subsequent_mask(tgt_mask)

        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output, decoder_mask)

        return decoder_output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(torch.float32)

    def add_positional_encoding(self, x, feature_size):
        even_i = torch.arange(0, feature_size, 2, device=x.device).float()
        denominator = torch.pow(10000, even_i / feature_size)
        position = torch.arange(x.shape[1], device=x.device).reshape(x.shape[1], 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        pe = torch.flatten(stacked, start_dim=1, end_dim=2)
        if feature_size % 2 == 1:
            pe = pe[:, :-1]

        return x + pe


class MultiTimeHorizonTransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers):
        super(MultiTimeHorizonTransformerModel, self).__init__()
        self.encoder_daily = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.encoder_weekly = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.encoder_yearly = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        
        self.decoder_daily = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.feature_size = feature_size
        self.decoder_weekly = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.feature_size = feature_size
        self.decoder_quaterly = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.feature_size = feature_size

        self.linear_layer = nn.Linear(3 * 2, 3)  
        
    def forward(self, src_daily, tgt_daily, src_weekly, tgt_weekly):
        tgt_mask = tgt_daily.shape[1] if len(tgt_daily.shape) == 3 else tgt_daily.shape[0]
        
        decoder_mask = self._generate_square_subsequent_mask(tgt_mask)

        encoder_output_daily = self.encoder_daily(src_daily)
        decoder_output_daily = self.decoder_daily(tgt_daily, encoder_output_daily, decoder_mask)

        encoder_output_weekly = self.encoder_weekly(src_weekly)
        decoder_output_weekly = self.decoder_weekly(tgt_weekly, encoder_output_weekly, decoder_mask)


        flat_daily = decoder_output_daily.view(-1, 3)
        flat_weekly = decoder_output_weekly.view(-1, 3)

        combined = torch.cat((flat_daily, flat_weekly), dim=1)

        output = self.linear_layer(combined)

        output = output.view(decoder_output_daily.shape[0], decoder_output_daily.shape[1], 3)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(torch.float32)

    def add_positional_encoding(self, x, feature_size):
        even_i = torch.arange(0, feature_size, 2, device=x.device).float()
        denominator = torch.pow(10000, even_i / feature_size)
        position = torch.arange(x.shape[1], device=x.device).reshape(x.shape[1], 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        pe = torch.flatten(stacked, start_dim=1, end_dim=2)
        if feature_size % 2 == 1:
            pe = pe[:, :-1]

        return x + pe

