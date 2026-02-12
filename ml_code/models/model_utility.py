import torch

def fit(model, loader, val_loader=None, epochs=10, learning_rate=0.01):
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(x)

            target = model.embedding_layer(x).to(device).float()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f'End of epoch {epoch}. MSE: {total_loss}')