import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers, device):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.device = device

    def forward(self, src, tgt):
        src_mask = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        tgt_mask = tgt.shape[1] if len(tgt.shape) == 3 else tgt.shape[0]
        
        encoder_mask = self._generate_square_subsequent_mask(src_mask, self.device)
        decoder_mask = self._generate_square_subsequent_mask(tgt_mask, self.device)
        # decoder_mask = self._generate_square_subsequent_mask(tgt_mask) & self._generate_future_mask(tgt_mask, src_mask)

        encoder_output = self.encoder(src, encoder_mask)
        decoder_output = self.decoder(tgt, encoder_output, decoder_mask)

        return decoder_output

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(torch.float32)
