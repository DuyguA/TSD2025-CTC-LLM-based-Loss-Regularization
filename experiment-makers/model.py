from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
import torch

class Wav2Vec2ForCTCWithIntermediateOutputs(Wav2Vec2ConformerForCTC):
    def forward(self, input_values, attention_mask=None, output_hidden_states=True, **kwargs):
        # Call the base class forward method
        outputs = super().forward(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        
        # Extract all hidden states (outputs of all encoder blocks)
        hidden_states = outputs.hidden_states  # List of (batch_size, seq_len, hidden_dim)
        
        # Get outputs from the 6th, 12th, 18th, and 24th blocks
        intermediate_outputs = {
            "block_6": hidden_states[6],
            "block_12": hidden_states[12],
            "block_18": hidden_states[18],
            "block_24": hidden_states[24],
        }
        
        return {
            "logits": outputs.logits,  # Final CTC logits
            "intermediate_outputs": intermediate_outputs,  # Intermediate layer outputs
        }

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ForCTCWithIntermediateOutputs.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

# Enable returning hidden states (important for intermediate outputs)
model.config.output_hidden_states = True

# Example input audio signal
input_audio = torch.randn(16000, )  # Replace with real audio data

# Preprocess audio
inputs = processor(input_audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Forward pass
outputs = model(**inputs)

# Access intermediate outputs
intermediate_outputs = outputs["intermediate_outputs"]
print("Output from block 6:", intermediate_outputs["block_6"].shape)
print("Output from block 12:", intermediate_outputs["block_12"].shape)
print("Output from block 18:", intermediate_outputs["block_18"].shape)
print("Output from block 24:", intermediate_outputs["block_24"].shape)

