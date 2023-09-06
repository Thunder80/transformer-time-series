import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers, max_len=5000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(feature_size, feature_size)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        self.max_len = max_len
        self.feature_size = feature_size
        
    def forward(self, src, tgt):
        src_size = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        tgt_size = tgt.shape[1] if len(tgt.shape) == 3 else tgt.shape[0]
        
        src_mask = self._generate_square_subsequent_mask(src_size)
        tgt_mask = self._generate_square_subsequent_mask(tgt_size)
        
        # Adding positional encoding directly
        # src = self.add_positional_encoding(src, self.feature_size)
        # tgt = self.add_positional_encoding(tgt, self.feature_size)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask)

        return decoder_output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(torch.float32)

    def add_positional_encoding(self, x, feature_size):
        even_i = torch.arange(0, feature_size, 2, device=x.device).float()
        denominator = torch.pow(10000, even_i / feature_size)
        position = torch.arange(self.max_len, device=x.device).reshape(self.max_len, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        pe = torch.flatten(stacked, start_dim=1, end_dim=2)
        if feature_size % 2 == 1:
            pe = pe[:, :-1]

        return x + pe
