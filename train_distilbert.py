import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # ==========================================
    # 1. HARDWARE CONSTRAINTS & HYPERPARAMETERS
    # ==========================================
    device = torch.device('cpu') # Enforce CPU
    BATCH_SIZE = 4               # Physical batch size (fits in 8GB RAM)
    ACCUMULATE_STEPS = 4         # Effective batch size = 16
    EPOCHS = 1                   # Kept to 1 for quick CPU demonstration

    print(f"--- Starting CPU-Optimized Training Pipeline ---")
    print(f"Device: {device} | Batch Size: {BATCH_SIZE} | Accumulation Steps: {ACCUMULATE_STEPS}")

    # ==========================================
    # 2. LOAD & PREPROCESS DATA (Automated)
    # ==========================================
    print("\nDownloading Tokenizer and IMDB Dataset (1% slice for CPU demo)...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load just 1% of the training data (~250 samples) so it runs fast on a CPU
    raw_dataset = load_dataset("imdb", split="train[:1%]")
    
    def preprocess_function(examples):
        # Truncate to 512 tokens to protect memory
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    # Tokenize and format for PyTorch
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 3. LOAD MODEL & APPLY LAYER FREEZING
    # ==========================================
    print("\nLoading DistilBERT Model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    print("Applying Layer Freezing to bottom 4 layers...")
    # Freeze embeddings
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
        
    # Freeze the first 4 transformer blocks
    for layer in model.distilbert.transformer.layer[:4]:
        for param in layer.parameters():
            param.requires_grad = False

    # Calculate parameter reduction
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)")

    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    # ==========================================
    # 4. TRAINING LOOP WITH GRADIENT ACCUMULATION
    # ==========================================
    print("\nStarting Training Loop...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device) # Note: datasets library uses 'label', not 'labels'

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Normalize the loss (Mathematical scaling for accumulation)
            loss = loss / ACCUMULATE_STEPS
            total_loss += loss.item()
            
            # Backward pass (compute gradients but DO NOT update weights yet)
            loss.backward()

            # Optimizer Step (Update weights only every ACCUMULATE_STEPS)
            if (step + 1) % ACCUMULATE_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item() * ACCUMULATE_STEPS:.4f}"})

    print("\n✅ CPU Training Demonstration Complete!")

if __name__ == "__main__":
    main()
