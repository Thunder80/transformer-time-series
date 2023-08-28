import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        
        self.month_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        # self.month_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.l1 = nn.Linear(4, 7)



    def forward(self, src, tgt, month_src):
        src_mask = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        tgt_mask = tgt.shape[1] if len(tgt.shape) == 3 else tgt.shape[0]
        
        encoder_mask = self._generate_square_subsequent_mask(src_mask)
        decoder_mask = self._generate_square_subsequent_mask(tgt_mask)
        # decoder_mask = self._generate_square_subsequent_mask(tgt_mask) & self._generate_future_mask(tgt_mask, src_mask)

        encoder_output = self.encoder(src, encoder_mask)
        decoder_output = self.decoder(tgt, encoder_output, decoder_mask)

        month_src_mask = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        month_encoder_mask = self._generate_square_subsequent_mask(month_encoder_mask)

        month_encoder_output = self.month_encoder(month_src, month_encoder_mask)

        combined_data = torch.cat((decoder_output, month_encoder_output), dim=1)
        output = self.l1(combined_data)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(torch.float32)

    def _generate_future_mask(self, tgt_size, src_size):
        mask = torch.ones(tgt_size, src_size)
        mask = torch.triu(mask, diagonal=src_size)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask.to(torch.float32)
