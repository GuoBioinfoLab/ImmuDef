from torch.utils.data import DataLoader, TensorDataset
from vae_model import BetaVAE
import torch.optim as optim
import argparse
import torch
import torch.nn.functional as F
import json


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a beta-VAE model.")
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=4.0, help='Beta value for KL divergence term')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of latent space')
    parser.add_argument('--input_dim', type=int, default=619, help='Dimensionality of input features')
    # Add encoder_layers argument, as decoder will be the reverse of it
    parser.add_argument('--encoder_layers', type=str, default='512 128 32',
                        help='Encoder layer sizes (space-separated)')
    # 添加 Beta 退火相关参数
    parser.add_argument('--anneal_beta', type=bool, default=False,
                        help='Enable beta annealing during training')
    parser.add_argument('--anneal_steps', type=int, default=10000,
                        help='Number of steps for linear annealing')
    parser.add_argument('--loss_fun', type=str, default='mse',
                        help='Loss function for training')
    parser.add_argument('--save_model', type=bool, default=True,
                        help='Save Pytorch Model')
    args = parser.parse_args()

    # 将 encoder_layers 字符串分割并转换为整数列表
    args.encoder_layers = list(map(int, args.encoder_layers.split()))
    # Decoder layers are the reverse of encoder layers
    decoder_layers = args.encoder_layers[::-1]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, and training setup
    model = BetaVAE(input_dim=619, latent_dim=2,
                    encoder_layers=args.encoder_layers,
                    decoder_layers=decoder_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 存储损失的字典
    his_list = []

    for epoch in range(1, args.epochs + 1):
        losses = {}
        if args.anneal_beta:
            scheduler = BetaScheduler(max_beta=args.beta, anneal_steps=args.epochs)
            schedule_beta = scheduler.get_beta(epoch)
            train_loss, train_detail_loss = train(model, dataloader, optimizer, device, args.beta, loss_fun=args.loss_fun)
            val_loss, val_detail_loss = validate(model, val_loader, device, args.beta, loss_fun=args.loss_fun)
        else:
            train_loss, train_detail_loss = train(model, dataloader, optimizer, device, args.beta, loss_fun=args.loss_fun)
            val_loss, val_detail_loss = validate(model, val_loader, device, args.beta, loss_fun=args.loss_fun)

        # 将当前的损失添加到字典中
        losses["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S"),
        losses["train_loss"] = {"training_loss": train_loss, "detail": train_detail_loss}
        losses["val_loss"] = {"validate_loss": val_loss, "detail": val_detail_loss}
        losses["epoch"] = epoch
        losses["params"] = vars(args)
        his_list.append(losses)
        # 打印每个 epoch 结束后的训练与验证损失
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 在训练完成后保存所有损失
    file_path = "losses.json"

    # 读取现有数据并追加新的内容
    try:
        with open(file_path, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # 直接追加新内容
    existing_data.extend(his_list)  # 或者使用 .extend(his_list)

    # 保存更新后的数据
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print("Training finished, losses saved to 'losses.pkl'")

    if args.save_model:
        model_filename = f"anneal_beta{args.anneal_beta}" \
                         f"_anneal_steps{args.anneal_steps}" \
                         f"enc{'_'.join(map(str, args.encoder_layers))}" \
                         f"_lr{args.lr}" \
                         f"_bs{args.batch_size}" \
                         f"_beta{args.beta}.pth"
        # 保证文件路径不被覆盖
        torch.save(model, model_filename)
