import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LLamaTokenizer

class CombinedCTCLMLoss(nn.Module):
  def __init__(self, llama_model, llama_tokenizer, lambda_llm=0.5):
      super(CombinedCTCLMLoss, self).__init__()
      self.lambda_llm = lambda_llm
      self.llama_model = llama_model
      self.llama_tokenizer = llama_tokenizer
      
  def forward(self, encoder_outputs, intermediate_outputs, input_lengths, target_texts, target_lengths):
      """
        Args:
            encoder_outputs: Final encoder outputs (B, T, C).
            intermediate_outputs: List of intermediate encoder outputs (from selected layers).
            input_lengths: Lengths of the audio inputs.
            target_texts: Ground truth transcriptions for the batch.
            target_lengths: Lengths of the target transcriptions.
            tokenizer: LLaMA tokenizer for tokenizing text.
            llm_model: LLaMA model for computing the LLM loss.

        Returns:
            total_loss: Combined loss (CTC + LLM).
        """
      tokenized_targets = self.llama_tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)
      target_indices = tokenized_targets["input_ids"].to(encoder_outputs.device)
      target_lengths = tokenized_targets["attention_mask"].sum(dim=1).to(encoder_outputs.device)  # Lengths

      encoder_outputs = encoder_outputs.permute(1, 0, 2) # Encoder outputs to (T, B, C) for ctc loss 

      # Reshape encoder logits for CTC loss
      log_probs = F.log_softmax(encoder_outputs, dim=-1)
      ctc_loss = self.ctc_loss_fn(log_probs, target_indices, input_lengths, target_lengths)


      # Now compute LLM loss for intermediate layers
      llm_losses = []
      for i,layer_output enumerate(intermediate_outputs):
          logits = F.log_softmax(layer_output, dim=-1)  # (B, T, Vocab)
          predicted_ids = torch.argmax(logits, dim=-1)  # (B, T)
          decoded_hypotheses = self.llama_tokenizer.batch_decode(predicted_ids,
                                                                 skip_special_tokens=True)


          llm_inputs = self.llm_tokenizer(decoded_hypotheses,
                                          return_tensors="pt", padding=True,
                                          truncation=True).to(self.llama_model.device)
          llm_labels = self.llm_tokenizer(target_texts, return_tensors="pt",
                                          padding=True,
                                          truncation=True).to(self.llama_model.device)
          llm_labels["input_ids"][llm_labels["attention_mask"] == 0] = -100

          # Compute LLM loss using the model's internal loss
          outputs = llm_model(**llm_inputs, labels=llm_labels["input_ids"])
          llm_losses.append(outputs.loss)

        # Average LLM losses across layers
      avg_llm_loss = torch.stack(llm_losses).mean()

        # ----- Step 3: Combine Losses -----
      total_loss = ctc_loss + self.lambda_llm * avg_llm_loss
      return total_loss


