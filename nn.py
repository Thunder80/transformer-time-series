import torch.nn as nn

class TransformerStockPrediction(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, num_heads, num_encoder_layers, num_decoder_layers):
        super(TransformerStockPrediction, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads),
            num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads),
            num_decoder_layers
        )
        self.input_embed = nn.Linear(input_dim, embed_size)
        self.output_embed = nn.Linear(embed_size, output_dim)

    def forward(self, src, tgt):
        src = self.input_embed(src)
        tgt = self.input_embed(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt[:-1, :, :], memory)
        output = self.output_embed(output)
        return output