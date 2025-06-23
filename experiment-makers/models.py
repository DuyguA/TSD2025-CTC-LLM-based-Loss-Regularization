import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerCTC

class IntermediateWav2Vec2CTC(nn.Module):
    def __init__(self, wav2vec_model, num_heads, llm_dim):
        super(IntermediateWav2VecCTC, self).__init__()
        
        self.num_heads = num_heads
        self.model = wav2vec_model
        self.model.config.output_hidden_states = True
        encoder_dim = self.model.hidden_size
        self.projection_layers = nn.ModuleList([nn.Linear(encoder_dim, llm_dim) for _ in range(self.num_heads)]) # 4 intermediate heads


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states

        inters = [6, 12, 18, 24]
        intermediate_outputs  = [self.projection_layers[i](hidden_states[inters[i]]) for i in range(self.num_heads)]

        return {
            "logits": outputs.logits, # Final CTC logits
            "intermediate_outputs": intermediate_outputs, #Intermediate logits
        }

