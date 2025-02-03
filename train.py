import warnings
from tqdm import tqdm
import os
from pathlib import Path

from contextlib import nullcontext
import torch
from torch.utils.tensorboard import SummaryWriter

from model import CustomBert
from dataloader import DataLoader
from config import get_config, get_weights_file_path

def load_labels(mode):
    if mode == 'train':
        with open('data/train_labels.txt', 'r') as f:
            labels = f.readlines()
    
    if mode == 'val':   
        with open('data/val_labels.txt', 'r') as f:
            labels = f.readlines()
    
    return labels

def load_dataset(mode):
    if mode == 'train':
        with open('data/train_dataset.txt', 'r') as f:
            dataset = f.readlines()

    if mode == 'val':
        with open('data/val_dataset.txt', 'r') as f:
            dataset = f.readlines()
    
    return dataset

def load_dataloaders(config):
    train_loader = DataLoader(batch_size=config['train_batch_size'], data = load_dataset('train'), labels = load_labels('train')) 
    val_loader = DataLoader(batch_size=config['val_batch_size'], data = load_dataset('val'), labels = load_labels('val'))
    return train_loader, val_loader

def get_model(config):
    model = CustomBert(**config)
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
        
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = load_dataloaders(config)
    model = get_model(config).to(device)

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    amp = nullcontext() if device == 'cpu' else nullcontext() if device == 'mps' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    if config['compile']:
        model = torch.compile(model) # requires
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    preload = config['preload']
    # model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # if model_filename:
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename)
    #     model.load_state_dict(state['model_state_dict'])
    #     initial_epoch = state['epoch'] + 1
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #     global_step = state['global_step']
    # else:
    #     print('No model to preload, starting from scratch')

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        train_dataloader.process_data()

        for batch in batch_iterator:

            encoder_input = batch['input_ids'].to(device) # (b, seq_len)
            encoder_mask = batch['attention_mask'].to(device) # (B, seq_len)
            labels = batch['labels'].to(device) 
            
            with amp:
            # Run the tensors through the encoder, decoder and the projection layer
                logits = model(encoder_input, encoder_mask, labels)
                token_logits, cls_token_logits, loss = logits['logits'], logits['cls_token_logits'], logits['Loss']
            
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # scaler.scale(loss).backward()
            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        # Need to update the validation function

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)