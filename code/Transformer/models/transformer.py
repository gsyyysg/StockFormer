from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import sys

sys.path.append('Transformer/')
from models.transformer_layer import EncoderLayer, DecoderLayer, Encoder, Decoder
from models.attn import FullAttention, AttentionLayer
from models.embed import DataEmbedding

class Transformer_base(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                d_model=128, n_heads=4, e_layers=2, d_layers=1, d_ff=256, 
                dropout=0.0, activation='gelu', output_attention=False):
        super(Transformer_base, self).__init__()

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        
        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)
    

        
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)

        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        
        output = self.projection_decoder(dec_out)


        return enc_out, dec_out, output


