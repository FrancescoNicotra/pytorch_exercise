from scripts.classes.dataset import Dataset
from scripts.classes.model import LSTMModel
from utils.utils import save_model, load_model, check_if_model_exists, plot_predictions, save_plot
from scripts.get_data import get_stock_data, save_data_to_csv
import torch
from torch.utils.data import DataLoader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fetch stock data
    stock_data = get_stock_data("AAPL", "2020-01-01", "2021-01-01")
    save_data_to_csv(stock_data, "AAPL_stock_data.csv")

    # Dataset & DataLoader
    dataset = Dataset("AAPL_stock_data.csv", sequence_length=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2).to(device)

    # Load existing model if available
    if check_if_model_exists("lstm_model.pth"):
        load_model(model, "lstm_model.pth")
        print("Model loaded from checkpoint.")
    else:
        print("No saved model found, training from scratch.")

    # Train model (semplice una epoca dimostrativa)
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = criterion(output, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_model(model, "lstm_model.pth")
    print("Model training complete and saved as lstm_model.pth")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            all_preds.append(output.cpu())
            all_targets.append(y_batch.cpu())

    # Concateno i tensori
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Plot
    plot_predictions(predictions, targets)
    save_plot(predictions, targets, "predictions_plot.png")


if __name__ == "__main__":
    main()
