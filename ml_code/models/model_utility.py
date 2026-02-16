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

def select_threshold(model, loader, alpha=0.95):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='none')
    model.to(device)
    model.eval()

    errors = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            output = model(x)

            target = model.embedding_layer(x).to(device).float()
            mse = criterion(output, target)

            mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)

            errors.append(mse_per_sample)

    errors = torch.cat(errors)

    threshold = torch.quantile(errors, alpha).item()

    del errors
        
    return threshold

def evaluate(model, test_loader, threshold):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss(reduction='none')
    model.to(device)
    model.eval()

    results = []

    total_mse = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            output = model(x)

            target = model.embedding_layer(x).to(device).float()
            mse = criterion(output, target)

            mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)

            total_mse += mse_per_sample.mean()

            predictions = (mse_per_sample >= threshold).int().cpu().tolist()
            results.extend(predictions)
    
    print(f'Average MSE: {total_mse / len(test_loader)}')

    return results