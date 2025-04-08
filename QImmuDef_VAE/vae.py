import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pickle
import pandas as pd
import numpy as np
import time
import json


class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers):
        super(BetaVAE, self).__init__()

        # Build encoder layers dynamically based on encoder_layers list
        encoder_layer_list = []
        current_input_dim = input_dim
        for units in encoder_layers:
            encoder_layer_list.append(nn.Linear(current_input_dim, units))
            encoder_layer_list.append(nn.ReLU())
            current_input_dim = units
        self.encoder = nn.Sequential(*encoder_layer_list)

        # Latent space
        self.mu_layer = nn.Linear(current_input_dim, latent_dim)
        self.logvar_layer = nn.Linear(current_input_dim, latent_dim)

        # Build decoder layers dynamically based on decoder_layers list
        decoder_layer_list = []
        current_input_dim = latent_dim
        for units in decoder_layers:
            decoder_layer_list.append(nn.Linear(current_input_dim, units))
            decoder_layer_list.append(nn.ReLU())
            current_input_dim = units
        decoder_layer_list.append(nn.Linear(current_input_dim, input_dim))  # Output layer without activation
        self.decoder = nn.Sequential(*decoder_layer_list)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

def beta_vae_loss(recon_x, x, mu, logvar, beta, loss_fun):
    # 计算重构损失 (均方误差)
    if loss_fun == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_fun == 'mae':
        loss_fn = nn.L1Loss(reduction='sum')
        recon_loss = loss_fn(recon_x, x)
    # 计算KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 返回重构损失、KL损失和总损失
    total_loss = recon_loss + beta * kl_loss
    return recon_loss, kl_loss, total_loss

class BetaScheduler:
    def __init__(self, max_beta, anneal_steps):
        self.max_beta = max_beta  # KL的最大权重
        self.anneal_steps = anneal_steps  # 退火步数

    def get_beta(self, step):
        return min(self.max_beta, step / self.anneal_steps)

def train(model, dataloader, optimizer, device, beta, loss_fun):
    model.train()
    total_train_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)  # 从 DataLoader 中提取数据并转移到设备
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)

        # 计算重构损失、KL损失和总损失
        recon_loss, kl_loss, total_loss = beta_vae_loss(recon_x, x, mu, logvar, beta, loss_fun)
        # 累加损失
        total_train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        total_loss.backward()
        optimizer.step()

    # 打印每个 epoch 的损失
    print(f"Train Loss: {total_train_loss / len(dataloader.dataset):.4f}, "
          f"Train Reconstruction Loss: {total_recon_loss / len(dataloader.dataset):.4f}, "
          f"Train KL Loss: {total_kl_loss / len(dataloader.dataset):.4f}")

    detail_loss = {"Train Reconstruction Loss": total_recon_loss / len(dataloader.dataset),
                   "Train KL Loss": total_kl_loss / len(dataloader.dataset)}

    return total_train_loss / len(dataloader.dataset), detail_loss


def validate(model, dataloader, device, beta, loss_fun):
    model.eval()
    total_val_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)  # 从 DataLoader 中提取数据并转移到设备
            recon_x, mu, logvar = model(x)

            # 计算重构损失、KL损失和总损失
            recon_loss, kl_loss, total_loss = beta_vae_loss(recon_x, x, mu, logvar, beta, loss_fun)

            # 累加损失
            total_val_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

    # 打印每个 epoch 的损失
    print(f"Validation Loss: {total_val_loss / len(dataloader.dataset):.4f}, "
          f"Validation Reconstruction Loss: {total_recon_loss / len(dataloader.dataset):.4f}, "
          f"Validation KL Loss: {total_kl_loss / len(dataloader.dataset):.4f}")

    detail_loss = {"Validation Reconstruction Loss": total_recon_loss / len(dataloader.dataset),
                   "Validation KL Loss": total_kl_loss / len(dataloader.dataset)}

    return total_val_loss / len(dataloader.dataset), detail_loss

# 保存训练过程的结果
def save_results(epoch, train_loss, val_loss, recon_loss, kl_loss, args):
    result = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": kl_loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": vars(args)
    }

    # 将结果存储到 JSON 文件
    with open('training_results.json', 'a') as f:
        json.dump(result, f, indent=4)
        f.write("\n")
