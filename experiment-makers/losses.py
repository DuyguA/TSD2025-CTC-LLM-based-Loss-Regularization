import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM


class ProjectedCTCLLMLoss(nn.Module):
    def __init__(self, tokenizer, llm_model, lambda_llm=0.5):
        """
        Args:
            lambda_llm: Weight for the LLM loss in the total loss.
        """
        super(ProjectedCTCLLMLoss, self).__init__()
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)  # CTC Loss
        self.lambda_llm = lambda_llm  # Weight for LLM loss
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def forward(self, encoder_outputs, intermediate_outputs, input_lengths, target_texts, target_lengths):
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
        llm_labels = self.tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(encoder_outputs.device)

        for layer_output in intermediate_outputs:  # Loop over intermediate layers
            # Pass projected intermediate outputs into LLM
            outputs = self.llm_model(inputs_embeds=layer_output, labels=llm_labels["input_ids"])
            
            # Extract loss from LLM
            llm_losses.append(outputs.loss)

        # Average LLM losses across layers
        avg_llm_loss = torch.stack(llm_losses).mean()

        # ----- Step 3: Combine Losses -----
        total_loss = ctc_loss + self.lambda_llm * avg_llm_loss
        return total_loss



