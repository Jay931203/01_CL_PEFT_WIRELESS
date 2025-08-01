# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:27:17 2024

This scripts performs the LWM inference on raw channel representations.

@author: Sadjad Alikhani
"""
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import visualize_embeddings
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score
import os
warnings.filterwarnings('ignore')
#%%
def lwm_inference(model, data, input_type="cls_emb", device="cpu", batch_size=64, visualization=False, task = 'LoS/NLoS Classification', mask= False, task_type = 'classification', test_type = 'backbone', labels=None, resume_path = None, visualization_method="t-sne"):
    if input_type == "raw":
        output_total = data
    else:
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        #inputs = []
        embeddings = []
        if resume_path is not None and os.path.exists(resume_path):
            model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"✅ Resuming from the resume_path: {resume_path}")
        else:
            print("🚀 Starting training from Latest")
        model.eval()
        with torch.no_grad():
            with tqdm(dataloader, desc="Inference", unit="batch") as t:
                for batch in t:

                    input_ids = batch[0].to(device)
                    #inputs.append(input_ids[:, 1:, :])
                    if (task == 'Embedding Regression'):
                        output = model(input_ids,  input_type=input_type, mask=mask)[0]
                    else:
                        output = model(input_ids)[0]
                    # if task == "Embedding Regression":
                    #     output = model.decoder(output) + model.decoder_bias

                    if input_type == "cls_emb":
                        batch_embeddings = output[:, 0, :]
                        embeddings.append(batch_embeddings)
                    elif input_type == "channel_emb":
                        if (test_type == 'backbone'):
                            batch_embeddings = output[:, 1:, :]
                        elif (test_type == 'full'):
                            batch_embeddings = output
                        embeddings.append(batch_embeddings)
        # print(f"ℹ️ input_type: {input_type}")
        # print(f"ℹ️ data shape: {data.shape}")
        # print(f"ℹ️ dataloader size: {len(dataloader)}")
        # inputs_total =  torch.cat(inputs, dim=0).float()
        labels = labels.to(device)
        # diff = torch.abs(inputs_total - labels)
        # print("(Masking Check) Mean absolute difference:", torch.mean(diff))

        output_total = torch.cat(embeddings, dim=0).float()
        if visualization:
            visualize_embeddings(output_total.view(output_total.size(0), -1),
                                 labels,
                                 task,
                                 method=visualization_method, 
                                 label="Embedding Space")
            visualize_embeddings(data.view(data.size(0), -1), 
                                 labels,
                                 task,
                                 method=visualization_method, 
                                 label="Original Space")
        metric = None
        if 0:
        #if test_type == 'backbone':
            if task_type == "regression":
                nmse = torch.mean(
                    torch.sum((output_total.view(output_total.size(0), -1) - labels.view(labels.size(0), -1)) ** 2, dim=1) /
                    (torch.sum(labels.view(labels.size(0), -1) ** 2, dim=1) + 1e-10)
                ).item()
                nmse2 = torch.mean(
                    torch.sum((output_total - labels) ** 2, dim=1) /
                    (torch.sum(labels**2, dim=1) + 1e-10)
                ).item()
                print(output_total.shape)
                print(output_total.view(output_total.size(0), -1).shape)
                print(labels.shape)
                print(labels.view(labels.size(0), -1).shape)
                print(f"[Evaluation] NMSE: {nmse:.6f}, {nmse2:.6f}")
                metric = nmse
                #sys.exit(1)
        else:
            if task_type == "classification":
                preds = output_total.argmax(dim=1)
                metric = f1_score(labels.cpu(), preds.cpu(), average='weighted')
                print(f"[Evaluation] F1-score: {metric:.4f}")
            else:
                print(output_total.shape)
                if (test_type == 'full'):
                    labels = labels.view(labels.size(0), -1)
                print(labels.shape)
                nmse = torch.mean(
                    torch.sum((output_total - labels.to(device)) ** 2, dim=1) /
                    (torch.sum(labels.to(device) ** 2, dim=1) + 1e-10)
                ).item()
                nmse2 = torch.mean(
                    torch.sum((output_total.view(output_total.size(0), -1) - labels.view(labels.size(0), -1)) ** 2, dim=1) /
                    (torch.sum(labels.view(labels.size(0), -1) ** 2, dim=1) + 1e-10)
                ).item()

                metric = nmse
                print(f"[Evaluation] NMSE: {nmse:.6f}, {nmse2:.6f}")
    return output_total, metric
