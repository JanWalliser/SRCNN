import csv
import matplotlib.pyplot as plt
# Plot-Skript zur Visualisierung der Verluste
def plot_loss(csv_path="losses.csv"):
    epochs = []
    train_losses = []
    test_losses = []

    # CSV-Datei lesen
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            train_losses.append(float(row["Train Loss"]))
            test_losses.append(float(row["Test Loss"]))

    # Verluste plotten
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, test_losses, label="Test Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")  # Plot als Bild speichern
    plt.show()

if __name__ == "__main__":
    plot_loss()
