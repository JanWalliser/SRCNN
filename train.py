import torch
import torch.nn as nn
import torch.optim as optim
from srcnn_model import SRCNN
from dataset import get_dataloader
from utils import save_model
from tqdm import tqdm  # Fortschrittsanzeige
from utils import save_model, load_model  # Lade- und Speicherfunktionen
import time
import csv
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(
    model, train_loader, test_loader, num_epochs=20, save_every=5, learning_rate=1e-4, checkpoint_path=None, csv_path="losses2.csv"
):
    
    if checkpoint_path:
        model = load_model(model, checkpoint_path)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # CSV-Datei vorbereiten
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Test Loss"])

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

        test_loss = evaluate_model(model, test_loader)
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
            f"Test Loss: {test_loss:.4f}, Time: {elapsed_time:.2f}s"
        )

        # Verluste in CSV-Datei speichern
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss / len(train_loader), test_loss])

        if (epoch + 1) % save_every == 0:
            save_model(model, epoch + 1)

if __name__ == "__main__":
    low_res_dir = "data/low_res"
    high_res_dir = "data/high_res"
    train_loader = get_dataloader(low_res_dir, high_res_dir, batch_size=32)
    test_loader = get_dataloader(low_res_dir, high_res_dir, batch_size=32)

    model = SRCNN().cuda()

    # Pfad zu deinem gespeicherten Modell (Ändere dies entsprechend)
    checkpoint_path = "checkpoints2/srcnn_epoch_365.pth"

    # Weiteres Training starten
    train_model(model, train_loader, test_loader, num_epochs=400, save_every=5, checkpoint_path = checkpoint_path)
