import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import numpy as np

import os, copy, json, csv
from tqdm import tqdm

from datasets import load_metric


from data_loaders import  data_loader, load_and_preprocess_dataset
from models import  IntermediateWav2Vec2CTC
from loss_functions import CombinedCTCLMLoss
from llm_models import llama_tokenizer, llm_model

class Trainer:
  def __init__(self, args):
    asr_device = "cuda:0"
    llm_device = "cuda:1"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    train_set = load_and_preprocess_dataset(args.dataset, "train")
    test_set = load_and_preprocess_dataset(args.dataset, "test")
    val_set = load_and_preprocess_dataset(args.dataset, "dev")

    self.train_loader = data_loader(train_set, batch_size=args.batch_size, shuffle=True)
    self.test_loader = data_loader(test_set, batch_size=args.batch_size, shuffle=True)
    self.valid_loader = data_loader(valid_set, batch_size=args.batch_size, shuffle=True)
    print('Dataset prep done!\n')

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found")

    print('Initializing model....')
    model = IntermediateWav2Vec2CTC(num_heads=args.num_heads, llm_dim=args.llm_dim)

    model = model.to(asr_device)
    params = model.parameters()
    self.scaler = GradScaler()

    self.optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    self.audio_device = asr_device
    self.llm_device = llm_device
    self.model = model

    self.loss_func = CombinedCTCLMLoss(llama_tokenizer, llama_model)
    self.wer_metric = load_metric("wer")

    self.args = args
    self.epoch_wers = []
    self.all_losses = []

  def train(self):
    best_epoch = 0
    print("First epoch will start soon")
    for epoch in range(self.args.epochs):
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
      loss = self.train_epoch()
      wer = self.eval()
      print(f'WER value: {wer:.3f}')
      self.epoch_wers.append(round(wer, 3)) # append epoch wers
      print("Training finished here are epoch WERs")
      with open(args.output_dir+"/epoch_acc.txt", "w") as ofile:
        for ewer in self.epoch_wers:
          ofile.write(str(ewer) + "\n")
      print("Now all losses")
      with open(args.output_dir+"/losses.txt", "w") as ofile:
        for eloss in self.all_losses:
          ofile.write(str(eloss) + "\n")
    self.final_test()

  def train_epoch(self):
    self.model.train()
    epoch_loss = 0
    loader = self.train_loader
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        self.optimizer.zero_grad()
        text = batch["text"]
        audio_lengths = batch["audio_lengths"]
        input_ids = batch["audio"].to(self.audio_device)
        attention_mask = batch["attention_mask"].to(self.audio_device)


        with autocast(device_type="cuda", dtype=torch.bfloat16):
          outputs = model(input_ids, attention_mask)
        logits, intermediate_outputs = outputs["logits"], outputs["intermediate_outputs"]

        loss = self.loss_func(logits, intermediate_outputs, audio_lengths, text)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        interval = max(len(loader) // 20, 1)
        if i % interval == 0 or i == len(loader) - 1:
            lloss = round(loss.item(), 3)
            #print(f'Batch: {i + 1}/{len(loader)}\ttotal loss: {loss.item():.3f}\temotion loss:{emotion_loss.item():.3f}\tpair loss:{pair_loss.item():.3f}')
            print(f'Batch: {i + 1}/{len(loader)}\ttotal loss: {lloss:.3f}')
            self.all_losses.append(lloss) # append epoch losses
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


  def eval(self):
    self.model.eval()  # Set the model to evaluation mode
    loader = self.valid_loader
    total_loss = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():  # Disable gradient computation for testing
        for batch in tqdm(loader, total=len(loader)):
            text = batch["text"]  # Ground truth text (list of strings)
            input_ids = batch["audio"].to(self.audio_device)  # Audio input IDs
            attention_mask = batch["attention_mask"].to(self.audio_device)  # Attention mask
            audio_lengths = batch["audio_lengths"]  # Audio lengths

            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(input_ids, attention_mask)
            
            logits = outputs["logits"]  # Final CTC logits
            
            # Compute loss
            loss = self.loss_func(logits, outputs["intermediate_outputs"], audio_lengths, text)
            total_loss += loss.item()

            # Decode logits into predicted text using processor
            pred_ids = torch.argmax(logits, dim=-1)
            predictions = self.processor.batch_decode(pred_ids, skip_special_tokens=True)  # Convert IDs to text

            # Append predictions and references for WER computation
            all_predictions.extend(predictions)
            all_references.extend(text)

    wer = self.wer_metric.compute(predictions=all_predictions, references=all_references)
    return wer

  def final_test(self):
    self.model.eval()  # Set the model to evaluation mode
    loader = self.valid_loader
    total_loss = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():  # Disable gradient computation for testing
        for batch in tqdm(loader, total=len(loader)):
            text = batch["text"]  # Ground truth text (list of strings)
            input_ids = batch["audio"].to(self.audio_device)  # Audio input IDs
            attention_mask = batch["attention_mask"].to(self.audio_device)  # Attention mask
            audio_lengths = batch["audio_lengths"]  # Audio lengths

            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(input_ids, attention_mask)
            
            logits = outputs["logits"]  # Final CTC logits
            
            # Compute loss
            loss = self.loss_func(logits, outputs["intermediate_outputs"], audio_lengths, text)
            total_loss += loss.item()

            # Decode logits into predicted text using processor
            pred_ids = torch.argmax(logits, dim=-1)
            predictions = self.processor.batch_decode(pred_ids, skip_special_tokens=True)  # Convert IDs to text

            # Append predictions and references for WER computation
            all_predictions.extend(predictions)
            all_references.extend(text)
    wer = self.wer_metric.compute(predictions=all_predictions, references=all_references)
    return wer



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="tedlium", help='Name of the dataset')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--output_dir', type=str, default="exp", help='Directory for saving the model')
    args = parser.parse_args()

    print(args)
    engine = Trainer(args)
    engine.train()
