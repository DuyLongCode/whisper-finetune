import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import WhisperTokenizer
inp=torch.randn(16,12,2000).to('cuda')

class MHSAExtBiMambaLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_state, d_conv, expand):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(d_model, num_heads)
        self.ext_bi_mamba = ExternalBidirectionalMambaLayer(d_model, d_state, d_conv, expand)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output, _ = self.mhsa(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # External Bidirectional Mamba
        mamba_output = self.ext_bi_mamba(x)
        x = x + self.dropout(mamba_output)
        x = self.layer_norm2(x)

        return x

class ExternalBidirectionalMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.output_linear = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # Forward direction
        forward_output = self.forward_mamba(x)
        
        # Backward direction
        backward_input = torch.flip(x, dims=[1])
        backward_output = self.backward_mamba(backward_input)
        backward_output = torch.flip(backward_output, dims=[1])
        
        # Concatenate forward and backward outputs
        combined_output = torch.cat((forward_output, backward_output), dim=-1)
        
        # Final linear layer
        output = self.output_linear(combined_output)
        
        return output
    
layer1=MHSAExtBiMambaLayer(inp.shape[2],16,16,16,16).to('cuda')
# layer2=ExternalBidirectionalMambaLayer(16,16,16,16,16)

out=layer1(inp)

print(out.shape)


import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ASRModel(nn.Module):
    def __init__(self, num_mel_bins, vocab_size, d_model, num_heads, d_state, d_conv, expand):
        super().__init__()
        self.mel_projection = nn.Linear(num_mel_bins, d_model)
        self.mhsa_extbimamba = MHSAExtBiMambaLayer(d_model, num_heads, d_state, d_conv, expand)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, mel_spectrogram):
        x = self.mel_projection(mel_spectrogram)
        x = self.mhsa_extbimamba(x)
        x = self.output_layer(x)
        return x

    
    def decode(self, mel_spectrogram, tokenizer):
        with torch.no_grad():
            logits = self.forward(mel_spectrogram)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Simple CTC decoding
            decoded_ids = []
            previous_id = 0  # Assuming 0 is both pad and blank token
            for id in predicted_ids[0]:  # Assuming batch size 1
                if id != 0 and id != previous_id:
                    decoded_ids.append(id.item())
                previous_id = id
            
            decoded_text = tokenizer.decode(decoded_ids)
        return decoded_text


num_mel_bins = 80  # Typical number of mel bins
vocab_size = 10000  # Adjust based on your tokenizer
d_model = 512
num_heads = 8
d_state = 16
d_conv = 4
expand = 2

model = ASRModel(num_mel_bins, vocab_size, d_model, num_heads, d_state, d_conv, expand).to('cuda')
model_name="vinai/PhoWhisper-large"
# Assuming you have a trained model, a tokenizer, and a Mel spectrogram
tokenizer =  WhisperTokenizer.from_pretrained(model_name,cache_dir="/media/sanslab/Data/DuyLong/whis",)
mel_spectrogram = torch.randn(1, 1000, num_mel_bins).to('cuda')  # Example shape: (batch_size, time_steps, num_mel_bins)

decoded_text = model.decode(mel_spectrogram, tokenizer)
print(decoded_text)





