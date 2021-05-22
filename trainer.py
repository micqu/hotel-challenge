import torch
import utility
import wandb

def train_model(device, model, optimizer, criterion, train_loader, valid_loader,
                scheduler, epochs, send_to_wandb: bool = False):
    # Run train loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_map = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.enable_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_map += utility.calculate_map(outputs, labels)
            
        model.eval()
        valid_loss = 0.0
        valid_map = 0.0
        for inputs, labels in valid_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * inputs.size(0)
            valid_map += utility.calculate_map(outputs, labels)

        train_loss /= len(train_loader.dataset)
        train_map /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        valid_map /= len(valid_loader.dataset)
        
        if send_to_wandb:     
            wandb.log({"train_loss": train_loss,
                    "train_map": train_map,
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                    "valid_map": valid_map})