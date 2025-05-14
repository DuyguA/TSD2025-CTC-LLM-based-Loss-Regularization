import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM

class CombinedCTCLLMLossWithIntermediate(nn.Module):
    def __init__(self, encoder_layers, lambda_llm=0.5):
        """
        Args:
            encoder_layers: List of layer indices to extract intermediate outputs.
            lambda_llm: Weight for the LLM loss in the total loss.
        """
        super(CombinedCTCLLMLossWithIntermediate, self).__init__()
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # CTC Loss
        self.lambda_llm = lambda_llm  # Weight for LLM loss
        self.encoder_layers = encoder_layers  # Layers to attach LLM loss
        self.projection_layers = nn.ModuleList([nn.Linear(encoder_dim, token_dim) for _ in encoder_layers])

    def forward(self, encoder_outputs, intermediate_outputs, input_lengths, target_texts, target_lengths, tokenizer, llm_model):
        """
        Args:
            encoder_outputs: Final encoder outputs (T, N, C).
            intermediate_outputs: List of intermediate encoder outputs (from selected layers).
            input_lengths: Lengths of the audio inputs.
            target_texts: Ground truth transcriptions for the batch.
            target_lengths: Lengths of the target transcriptions.
            tokenizer: LLaMA tokenizer for tokenizing text.
            llm_model: LLaMA model for computing the LLM loss.

        Returns:
            total_loss: Combined loss (CTC + LLM).
        """
        # ----- Step 1: Compute CTC Loss -----
        # Convert target texts to indices for CTC loss
        target_indices = [torch.tensor(tokenizer.encode(text)) for text in target_texts]
        target_indices = torch.cat(target_indices).to(encoder_outputs.device)
        target_lengths = torch.tensor([len(text) for text in target_texts]).to(encoder_outputs.device)

        # Reshape encoder logits (T, N, C) for CTC loss
        log_probs = F.log_softmax(encoder_outputs, dim=-1)
        ctc_loss = self.ctc_loss_fn(log_probs, target_indices, input_lengths, target_lengths)

        # ----- Step 2: Compute LLM Loss for Intermediate Layers -----
        llm_losses = []
        for i, (layer_output, projection_layer) in enumerate(zip(intermediate_outputs, self.projection_layers)):
            # Project intermediate features to token space
            projected_output = projection_layer(layer_output)  # (T, N, token_dim)

            # Greedy decoding for intermediate layer outputs
            decoded_hypotheses = []
            for j in range(projected_output.size(1)):  # Iterate over the batch
                logits = F.log_softmax(projected_output[:, j, :], dim=-1)  # (T, token_dim)
                predicted_ids = torch.argmax(logits, dim=-1)  # Greedy decoding
                predicted_text = tokenizer.decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
                decoded_hypotheses.append(predicted_text)

            # Tokenize hypotheses and compute LLM loss
            llm_inputs = tokenizer(decoded_hypotheses, return_tensors="pt", padding=True, truncation=True).to(encoder_outputs.device)
            llm_labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(encoder_outputs.device)

            # Compute LLM loss using the model's internal loss
            outputs = llm_model(**llm_inputs, labels=llm_labels["input_ids"])
            llm_losses.append(outputs.loss)

        # Average LLM losses across layers
        avg_llm_loss = torch.stack(llm_losses).mean()

        # ----- Step 3: Combine Losses -----
        total_loss = ctc_loss + self.lambda_llm * avg_llm_loss
        return total_loss



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM

class ProjectedCTCLLMLoss(nn.Module):
    def __init__(self, encoder_dim, token_dim, intermediate_layers, lambda_llm=0.5):
        """
        Args:
            encoder_dim: Dimension of the encoder's output features.
            token_dim: Dimension of the LLM's token embeddings.
            intermediate_layers: List of encoder layers to attach projection+LLM loss.
            lambda_llm: Weight for the LLM loss in the total loss.
        """
        super(ProjectedCTCLLMLoss, self).__init__()
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # CTC Loss
        self.lambda_llm = lambda_llm  # Weight for LLM loss
        self.intermediate_layers = intermediate_layers  # Intermediate layers for projection
        self.projection_layers = nn.ModuleList([nn.Linear(encoder_dim, token_dim) for _ in intermediate_layers])

    def forward(self, encoder_outputs, intermediate_outputs, input_lengths, target_texts, target_lengths, tokenizer, llm_model):
        """
        Args:
            encoder_outputs: Final encoder outputs (T, N, C).
            intermediate_outputs: List of intermediate encoder outputs (from selected layers).
            input_lengths: Lengths of the audio inputs.
            target_texts: Ground truth transcriptions for the batch.
            target_lengths: Lengths of the target transcriptions.
            tokenizer: LLaMA tokenizer for tokenizing text.
            llm_model: LLaMA model for computing the LLM loss.

        Returns:
            total_loss: Combined loss (CTC + LLM).
        """
        # ----- Step 1: Compute CTC Loss -----
        # Convert target texts to indices for CTC loss
        target_indices = [torch.tensor(tokenizer.encode(text)) for text in target_texts]
        target_indices = torch.cat(target_indices).to(encoder_outputs.device)
        target_lengths = torch.tensor([len(text) for text in target_texts]).to(encoder_outputs.device)

        # Reshape encoder logits (T, N, C) for CTC loss
        log_probs = F.log_softmax(encoder_outputs, dim=-1)  # Final encoder log probabilities
        ctc_loss = self.ctc_loss_fn(log_probs, target_indices, input_lengths, target_lengths)

        # ----- Step 2: Compute LLM Loss for Intermediate Layers -----
        llm_losses = []
        for i, (layer_output, projection_layer) in enumerate(zip(intermediate_outputs, self.projection_layers)):
            # Project intermediate features to token embedding space
            projected_output = projection_layer(layer_output)  # (T, N, token_dim)

            # Convert projected output to token probabilities (greedy decoding)
            decoded_hypotheses = []
            for j in range(projected_output.size(1)):  # Iterate over the batch
                token_probs = F.log_softmax(projected_output[:, j, :], dim=-1)
                predicted_ids = torch.argmax(token_probs, dim=-1)  # Greedy decoding
                predicted_text = tokenizer.decode(predicted_ids.cpu().numpy(), skip_special_tokens=True)
                decoded_hypotheses.append(predicted_text)

            # Tokenize hypotheses and compute LLM loss
            llm_inputs = tokenizer(decoded_hypotheses, return_tensors="pt", padding=True, truncation=True).to(encoder_outputs.device)
            llm_labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(encoder_outputs.device)

            # Compute LLM loss using the model's internal loss
            outputs = llm_model(**llm_inputs, labels=llm_labels["input_ids"])
            llm_losses.append(outputs.loss)

        # Average LLM losses across layers
        avg_llm_loss = torch.stack(llm_losses).mean()

        # ----- Step 3: Combine Losses -----
        total_loss = ctc_loss + self.lambda_llm * avg_llm_loss
        return total_loss



Total Loss = CTC Loss + λ_early * LLM Loss (early layers) + λ_late * LLM Loss (late layers)


# Define the projection layers for intermediate outputs
projection_layers = nn.ModuleList([
    nn.Linear(encoder_dim, token_dim) for _ in range(4)  # 4 intermediate heads
])

# Extract outputs from blocks 4, 8, 12, 16
intermediate_outputs = [block4_output, block8_output, block12_output, block16_output]

# Compute LLM loss for each intermediate output
llm_losses = []
for i, (layer_output, projection_layer) in enumerate(zip(intermediate_outputs, projection_layers)):
    projected_output = projection_layer(layer_output)  # Map to token embedding space
    # Decode and compute LLM loss as before
    ...


