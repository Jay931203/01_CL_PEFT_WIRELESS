# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:27:17 2024

This scripts performs the LWM inference on raw channel representations.

@author: Sadjad Alikhani
"""
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

        if test_type == 'backbone':
            if task_type == "regression":
                nmse = torch.mean(
                    torch.sum((output_total - labels.to(device)) ** 2, dim=1) /
                    (torch.sum(labels.to(device) ** 2, dim=1) + 1e-10)
                ).item()
                print(f"[Evaluation] NMSE: {nmse:.6f}")
        else:
            if task_type == "classification":
                preds = output_total.argmax(dim=1)
                print(f"[Evaluation] F1-score: {f1_score(labels.cpu(), preds.cpu(), average='weighted'):.4f}")
            else:
                if (test_type == 'full'):
                    labels = labels.view(labels.size(0), -1)
                nmse = torch.mean(
                    torch.sum((output_total - labels.to(device)) ** 2, dim=1) /
                    (torch.sum(labels.to(device) ** 2, dim=1) + 1e-10)
                ).item()
                print(f"[Evaluation] NMSE: {nmse:.6f}")
    return output_total
