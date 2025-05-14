[200~import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class DistilledRoberta(nn.Module):
    def __init__(self, num_labels, teacher_hidden_dim=2048):
        super(DistilledRoberta, self).__init__()
        
        self.student_model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels
        )
        student_hidden_dim = self.student_model.config.hidden_size
        # Add a projection layer to align teacher and student representations
        self.projection_layer = nn.Linear(teacher_hidden_dim, student_hidden_dim)  # Teacher -> Student dim

    def forward(self, input_ids, attention_mask, teacher_hidden=None):
        """
        Forward pass for the distillation model.
        
        Args:
        - input_ids: Input token IDs for the student model.
        - attention_mask: Attention mask for the student model.
        - labels: Ground truth labels for cross-entropy loss.
        - teacher_hidden: Teacher's representation (e.g., last token embedding) for MSE loss.
        
        Returns:
        - logits: Student's output logits for classification.
        - student_hidden: [CLS] token embedding from the student model.
        - projected_teacher_hidden: <eos> token embedding from teacher model, projected to hidden dim of student model
        """
        # Forward pass through the student model
        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Extract logits and the [CLS] token embedding
        student_logits = outputs.logits
        student_hidden = outputs.hidden_states[-1][:, 0, :]  # CLS token embedding
        
        # If teacher_hidden is provided, project it to the student's hidden dimension
        projected_teacher_hidden = None
        if teacher_hidden is not None:
            projected_teacher_hidden = self.projection_layer(teacher_hidden)
        else:
            projected_teacher_hidden = None
        
        return student_logits, student_hidden, projected_teacher_hidden

