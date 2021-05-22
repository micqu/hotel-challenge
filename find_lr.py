import math
import utility
import torch
import data_loader as dl
import matplotlib.pyplot as plt

USE_FEATURE_EXTRACT = True
BATCH_SIZE = 64

def main():
    logs, losses = find_lr()
    losses_np = [loss.detach().cpu().numpy() for loss in losses]
    plt.plot(logs, losses_np)
    plt.xscale('log', base=2)
    plt.xlabel('learning rate (log scale)')
    plt.ylabel('loss')
    plt.show()

def find_lr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build dataset
    train_loader, _, _, n_classes = dl.get_train_valid_loader(
        data_dir='data/train_images',
        meta_data_file='data/train.csv',
        batch_size=BATCH_SIZE,
        augment=True,
        random_seed=0
    )
    
    # Make resnet
    model = utility.initialize_resnet(n_classes, 'resnet18',
                                      feature_extract=USE_FEATURE_EXTRACT)
    model = model.to(device)
    
    params_to_update = model.parameters()
    if USE_FEATURE_EXTRACT:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    
    init_value=1e-8
    final_value=100.0
    
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_update, lr=lr)
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batch_num += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            print("Loss exploded")
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values
        losses.append(loss)
        log_lrs.append(lr)
        
        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
        
    return log_lrs[10:-5], losses[10:-5]

if __name__ == "__main__":
    main()