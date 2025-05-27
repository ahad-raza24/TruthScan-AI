import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import os
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import time
from datetime import datetime
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Utils:
    """Utility class for various helper functions"""
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory to ensure maximum available space"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def set_seed(seed=42):
        """Set up deterministic behavior for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def save_object(obj, path):
        """Save an object to disk using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_object(path):
        """Load an object from disk using pickle"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def plot_training_history(history, model_name, save_dir, display=True):
        """Plot and save training/validation metrics history"""
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_training_history.png", dpi=300, bbox_inches='tight')
        
        if display:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_confusion_matrix(labels, preds, model_name, save_dir, display=True):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        
        if display:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_metrics_comparison(results_dict, save_dir, display=True):
        """Plot and save comparison of model metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(results_dict.keys())
        
        # Create dataframe for metrics
        data = []
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(results_dict[model][metric])
            data.append(row)
        
        df = pd.DataFrame(data, columns=['Model'] + metrics)
        
        # Plot metrics comparison
        plt.figure(figsize=(10, 6))
        df_plot = df.set_index('Model')
        ax = df_plot.plot(kind='bar', rot=0, width=0.7)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_metrics_comparison.png", dpi=300, bbox_inches='tight')
        
        if display:
            plt.show()
        else:
            plt.close()
        
        # Save comparison to CSV
        df.to_csv(f"{save_dir}/model_metrics_comparison.csv", index=False)
        return df


class CombinedFeatureDataset(Dataset):
    """Dataset that combines embeddings, TF-IDF, and other features"""
    
    def __init__(self, embeddings, tfidf, features, labels):
        self.embeddings = embeddings
        self.tfidf = tfidf
        self.features = np.asarray(features)
        self.labels = labels
        
        # Check and handle NaN values
        self._check_and_handle_nans()
    
    def _check_and_handle_nans(self):
        """Check for NaN values and replace them with zeros"""
        if np.any(np.isnan(self.embeddings)):
            print("Warning: NaN values detected in embeddings")
            self.embeddings = np.nan_to_num(self.embeddings, nan=0.0)
        if np.any(np.isnan(self.features)):
            print("Warning: NaN values detected in features")
            self.features = np.nan_to_num(self.features, nan=0.0)
        if np.any(np.isnan(self.labels)):
            print("Warning: NaN values detected in labels")
            self.labels = np.nan_to_num(self.labels, nan=0.0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float)
        
        # Handle sparse TF-IDF matrices
        tfidf_row = self.tfidf[idx]
        if sparse.issparse(tfidf_row):
            tfidf = torch.tensor(tfidf_row.toarray().flatten(), dtype=torch.float)
        else:
            tfidf = torch.tensor(tfidf_row.flatten(), dtype=torch.float)
        
        feat = torch.tensor(self.features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return {
            'embeddings': emb,
            'tfidf': tfidf,
            'features': feat,
            'label': label
        }
class EmbeddingBranch(nn.Module):
    """Embedding processing branch of the neural network"""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.3):
        super(EmbeddingBranch, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)


class TfidfBranch(nn.Module):
    """TF-IDF processing branch of the neural network"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=128, dropout=0.3):
        super(TfidfBranch, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)


class FeatureBranch(nn.Module):
    """Feature processing branch of the neural network"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, dropout=0.3):
        super(FeatureBranch, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)

class FakeNewsDetector(nn.Module):
    """Model that combines embeddings, TF-IDF, and other features"""
    
    def __init__(self, embedding_dim=384, tfidf_dim=5000, feature_dim=16, dropout=0.3):
        super(FakeNewsDetector, self).__init__()
        
        self.embedding_branch = EmbeddingBranch(embedding_dim, dropout=dropout)
        self.tfidf_branch = TfidfBranch(tfidf_dim, dropout=dropout)
        self.feature_branch = FeatureBranch(feature_dim, dropout=dropout)
        
        # Combined input dimension from all branches
        combined_dim = 128 + 128 + 32  # outputs from each branch
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, embeddings, tfidf, features):
        emb_output = self.embedding_branch(embeddings)
        tfidf_output = self.tfidf_branch(tfidf)
        feat_output = self.feature_branch(features)
        combined = torch.cat((emb_output, tfidf_output, feat_output), dim=1)
        return self.classifier(combined)
    
    def summary(self):
        """Return a summary of the model architecture"""
        embedding_params = sum(p.numel() for p in self.embedding_branch.parameters())
        tfidf_params = sum(p.numel() for p in self.tfidf_branch.parameters())
        feature_params = sum(p.numel() for p in self.feature_branch.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = embedding_params + tfidf_params + feature_params + classifier_params
        
        summary = {
            'Embedding Branch Parameters': embedding_params,
            'TF-IDF Branch Parameters': tfidf_params,
            'Feature Branch Parameters': feature_params,
            'Classifier Parameters': classifier_params,
            'Total Parameters': total_params
        }
        return summary

class DataManager:
    """Class for handling data loading and preparation"""
    
    def __init__(self, base_dir='/kaggle/working'):
        self.base_dir = base_dir
        self.processed_data_path = f'{base_dir}/data/processed'
        self.data = None
        self.feature_cols = [
            'text_length', 'word_count', 'sentiment_score', 'subjectivity_score',
            'clickbait_score', 'emotional_score', 'exaggeration_score', 'inconsistency_score',
            'avg_word_length', 'avg_sentence_length', 'readability_score', 'unique_word_ratio',
            'title_sentiment_score', 'title_subjectivity_score', 'sentiment_discrepancy',
            'source_credibility'
        ]
    
    def load_data(self):
        """Load processed data from the preprocessing output"""
        print("Loading preprocessed data...")
        
        try:
            with open(f'{self.processed_data_path}/train_feature_store.pkl', 'rb') as f:
                train_store = pickle.load(f)
            with open(f'{self.processed_data_path}/val_feature_store.pkl', 'rb') as f:
                val_store = pickle.load(f)
            with open(f'{self.processed_data_path}/test_feature_store.pkl', 'rb') as f:
                test_store = pickle.load(f)
            
            # Load DeBERTa embeddings
            train_deberta_embeddings = np.load(train_store['embeddings_path'])
            val_deberta_embeddings = np.load(val_store['embeddings_path'])
            test_deberta_embeddings = np.load(test_store['embeddings_path'])
            
            # Load RoBERTa embeddings
            train_roberta_embeddings = np.load(train_store['roberta_embeddings_path'])
            val_roberta_embeddings = np.load(val_store['roberta_embeddings_path'])
            test_roberta_embeddings = np.load(test_store['roberta_embeddings_path'])
            
            # Load TF-IDF features
            train_tfidf = sparse.load_npz(train_store['tfidf_path'])
            val_tfidf = sparse.load_npz(val_store['tfidf_path'])
            test_tfidf = sparse.load_npz(test_store['tfidf_path'])
            
            # Check for missing feature columns
            missing_cols = [col for col in self.feature_cols if col not in train_store['features'].columns]
            if missing_cols:
                print(f"Warning: Missing feature columns in train_store: {missing_cols}")
                self.feature_cols = [col for col in self.feature_cols if col in train_store['features'].columns]
            
            # Extract features and labels
            train_features = train_store['features'][self.feature_cols].values
            val_features = val_store['features'][self.feature_cols].values
            test_features = test_store['features'][self.feature_cols].values
            
            train_labels = train_store['features']['binary_label'].values
            val_labels = val_store['features']['binary_label'].values
            test_labels = test_store['features']['binary_label'].values
            
            print(f"Loaded train samples: {len(train_labels)}")
            print(f"Loaded validation samples: {len(val_labels)}")
            print(f"Loaded test samples: {len(test_labels)}")
            print(f"Train DeBERTa embeddings shape: {train_deberta_embeddings.shape}")
            print(f"Train RoBERTa embeddings shape: {train_roberta_embeddings.shape}")
            print(f"Train TF-IDF shape: {train_tfidf.shape}")
            print(f"Train features shape: {train_features.shape}")
            
            self.data = {
                'train': {
                    'deberta_embeddings': train_deberta_embeddings,
                    'roberta_embeddings': train_roberta_embeddings,
                    'tfidf': train_tfidf,
                    'features': train_features,
                    'labels': train_labels
                },
                'val': {
                    'deberta_embeddings': val_deberta_embeddings,
                    'roberta_embeddings': val_roberta_embeddings,
                    'tfidf': val_tfidf,
                    'features': val_features,
                    'labels': val_labels
                },
                'test': {
                    'deberta_embeddings': test_deberta_embeddings,
                    'roberta_embeddings': test_roberta_embeddings,
                    'tfidf': test_tfidf,
                    'features': test_features,
                    'labels': test_labels
                }
            }
            
            return self.data
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Make sure preprocessing script ran successfully.")
            raise
        except KeyError as e:
            print(f"Error loading data: Missing key {e}. Check feature store contents.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_dataloaders(self, model_type, batch_size=64):
        """Create train, validation, and test dataloaders for specific model type"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        num_data_workers = 2 if torch.cuda.is_available() else 0
        print(f"Creating {model_type} DataLoaders with {num_data_workers} workers.")
        
        # Select the right embeddings based on model type
        embedding_key = f"{model_type.lower()}_embeddings"
        
        # Create datasets
        train_dataset = CombinedFeatureDataset(
            self.data['train'][embedding_key], self.data['train']['tfidf'],
            self.data['train']['features'], self.data['train']['labels']
        )
        val_dataset = CombinedFeatureDataset(
            self.data['val'][embedding_key], self.data['val']['tfidf'],
            self.data['val']['features'], self.data['val']['labels']
        )
        test_dataset = CombinedFeatureDataset(
            self.data['test'][embedding_key], self.data['test']['tfidf'],
            self.data['test']['features'], self.data['test']['labels']
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  pin_memory=True, num_workers=num_data_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, 
                               pin_memory=True, num_workers=num_data_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, 
                                pin_memory=True, num_workers=num_data_workers)
        
        return train_loader, val_loader, test_loader
    
    def validate_data_dimensions(self):
        """Validate dimensions of loaded data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Validating data dimensions...")
        try:
            assert self.data['train']['deberta_embeddings'].shape[0] == len(self.data['train']['labels']), "Mismatch in train data dimensions"
            assert self.data['val']['deberta_embeddings'].shape[0] == len(self.data['val']['labels']), "Mismatch in validation data dimensions" 
            assert self.data['test']['deberta_embeddings'].shape[0] == len(self.data['test']['labels']), "Mismatch in test data dimensions"
            print("Data dimensions validated successfully")
            return True
        except AssertionError as e:
            print(f"Data validation error: {e}")
            return False
    
    def check_for_nan_values(self):
        """Check for NaN values in data and replace them with zeros"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Checking for NaN values in data...")
        for data_type in ['train', 'val', 'test']:
            has_nan = False
            for key in ['deberta_embeddings', 'roberta_embeddings', 'features', 'labels']:
                if np.any(np.isnan(self.data[data_type][key])):
                    print(f"Warning: NaN values found in {data_type} {key}")
                    has_nan = True
                    # Replace NaNs with zeros for training stability
                    self.data[data_type][key] = np.nan_to_num(self.data[data_type][key], nan=0.0)
            if not has_nan:
                print(f"No NaN values found in {data_type} data")
        return self.data

class ModelTrainer:
    """Class for training and evaluating models"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train(self, model, train_loader, val_loader, epochs=4, lr=1e-4, warmup_steps=0, 
              gradient_accumulation_steps=1, model_dir='.', model_name='model'):
        """Train the model with learning rate scheduler and mixed precision"""
        model = model.to(self.device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        num_training_steps = len(train_loader) // gradient_accumulation_steps * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        best_val_acc = -1
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Mixed precision enabled: {use_amp}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            model.train()
            epoch_train_loss = 0
            train_preds = []
            train_labels = []
            progress_bar_train = tqdm(train_loader, desc="Training", leave=False)
            
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(progress_bar_train):
                embeddings = batch['embeddings'].to(self.device)
                tfidf = batch['tfidf'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(embeddings, tfidf, features)
                    loss = criterion(outputs.squeeze(1), labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping...")
                    continue
                
                # Collect predictions for accuracy
                probs = torch.sigmoid(outputs.squeeze(1)).detach().cpu().numpy()
                preds = (probs > 0.5).astype(float)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
                
                loss_scaled = loss / gradient_accumulation_steps
                scaler.scale(loss_scaled).backward()
                
                epoch_train_loss += loss.item()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar_train.set_postfix({'loss': f'{epoch_train_loss / (batch_idx + 1):.4f}'})
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # Validation phase
            val_results = self.evaluate(model, val_loader, criterion, eval_type="Validation")
            history['val_loss'].append(val_results['loss'])
            history['val_accuracy'].append(val_results['accuracy'])
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds")
            print(f"  Training Loss: {avg_train_loss:.4f} | Training Accuracy: {train_accuracy:.4f}")
            print(f"  Validation Loss: {val_results['loss']:.4f} | Validation Accuracy: {val_results['accuracy']:.4f}")
            
            # Save best model based on validation accuracy
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                best_model_path = f"{model_dir}/{model_name}_best.pt"
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best {model_name} saved to {best_model_path} with Val Acc: {best_val_acc:.4f}")
            
            Utils.clear_gpu_memory()
        
        # Save final model
        final_model_path = f"{model_dir}/{model_name}_final.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"  Final {model_name} saved to {final_model_path}")
        
        return model, history
    
    def evaluate(self, model, data_loader, criterion=None, eval_type="Test", model_name='model'):
        """Evaluate the model on test data"""
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        
        model = model.to(self.device)
        model.eval()
        
        eval_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        progress_bar_eval = tqdm(data_loader, desc=f"Evaluating ({eval_type})", leave=False)
        with torch.no_grad():
            for batch in progress_bar_eval:
                embeddings = batch['embeddings'].to(self.device)
                tfidf = batch['tfidf'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(embeddings, tfidf, features)
                loss = criterion(outputs.squeeze(1), labels)
                eval_loss += loss.item()
                
                probs = torch.sigmoid(outputs.squeeze(1)).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar_eval.set_postfix({f'{eval_type.lower()}_loss': f'{eval_loss / (len(all_preds) / data_loader.batch_size):.4f}'})
        
        avg_eval_loss = eval_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"\n{eval_type} Results for {model_name}:")
        print(f"  {eval_type} Loss: {avg_eval_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return {
            'loss': avg_eval_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'preds': np.array(all_preds),
            'probs': np.array(all_probs),
            'labels': np.array(all_labels)
        }
class FakeNewsTrainingPipeline:
    """Main class to orchestrate the training pipeline"""

    def __init__(self, base_dir='/kaggle/working', batch_size=64, epochs=5, 
                 lr=1e-4, gradient_accumulation_steps=1, dropout=0.3, display_plots=True):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dropout = dropout
        self.display_plots = display_plots

        # Create timestamped run directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_model_dir = f'{base_dir}/models'
        self.run_dir = f'{self.base_model_dir}/run_{self.timestamp}'
        self.deberta_model_dir = f'{self.run_dir}/deberta'
        self.roberta_model_dir = f'{self.run_dir}/roberta'

        # Create directories
        os.makedirs(self.deberta_model_dir, exist_ok=True)
        os.makedirs(self.roberta_model_dir, exist_ok=True)

        # Initialize components
        self.data_manager = DataManager(base_dir=base_dir)
        self.trainer = ModelTrainer()

        # Results storage
        self.results = {
            'deberta': {
                'model': None,
                'history': None,
                'test_results': None
            },
            'roberta': {
                'model': None,
                'history': None,
                'test_results': None
            }
        }

        print(f"Starting training run: {self.timestamp}")
        print(f"Configuration - Batch Size: {batch_size}, Epochs: {epochs}, LR: {lr}, " 
              f"Grad Accum: {gradient_accumulation_steps}, Dropout: {dropout}")
        print(f"DeBERTa results will be saved to: {self.deberta_model_dir}")
        print(f"RoBERTa results will be saved to: {self.roberta_model_dir}")
    
    def run(self):
        """Execute the complete training pipeline for DeBERTa and RoBERTa models"""
        # Set up training environment
        Utils.set_seed(42)
        Utils.clear_gpu_memory()
        
        # Load and prepare data
        try:
            data = self.data_manager.load_data()
            if not data:
                raise ValueError("Data loading failed")
                
            # Validate data
            self.data_manager.validate_data_dimensions()
            self.data_manager.check_for_nan_values()
            
            # Train DeBERTa model
            self._train_deberta_model()
            
            # Clear memory before training RoBERTa model
            Utils.clear_gpu_memory()
            
            # Train RoBERTa model
            self._train_roberta_model()
            
            # Compare results
            self._compare_models()
            
            # Visualize results
            self._visualize_results()
            
            return self.results
            
        except Exception as e:
            print(f"Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def _train_deberta_model(self):
        """Train and evaluate the DeBERTa model"""
        try:
            print("\n" + "="*50)
            print("Training DeBERTa Model")
            print("="*50)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = self.data_manager.create_dataloaders(
                model_type='deberta', batch_size=self.batch_size
            )
            
            # Get model dimensions
            deberta_embedding_dim = self.data_manager.data['train']['deberta_embeddings'].shape[1]
            tfidf_dim = self.data_manager.data['train']['tfidf'].shape[1]
            feature_dim = self.data_manager.data['train']['features'].shape[1]
            
            print(f"DeBERTa Model dimensions - Embedding: {deberta_embedding_dim}, "
                  f"TF-IDF: {tfidf_dim}, Features: {feature_dim}")
            
            # Create model
            deberta_model = FakeNewsDetector(
                embedding_dim=deberta_embedding_dim, tfidf_dim=tfidf_dim,
                feature_dim=feature_dim, dropout=self.dropout
            )
            
            # Print model summary
            model_summary = deberta_model.summary()
            print("\nDeBERTa Model Summary:")
            for k, v in model_summary.items():
                print(f"{k}: {v:,}")
            
            # Calculate warmup steps
            num_training_steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
            total_training_steps = num_training_steps_per_epoch * self.epochs
            warmup_steps = int(total_training_steps * 0.1)
            print(f"DeBERTa - Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}")
            
            # Train model
            trained_deberta_model, deberta_history = self.trainer.train(
                deberta_model, train_loader, val_loader,
                epochs=self.epochs, lr=self.lr, warmup_steps=warmup_steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                model_dir=self.deberta_model_dir, model_name='deberta'
            )
            
            # Evaluate model on test set
            criterion = nn.BCEWithLogitsLoss()
            deberta_test_results = self.trainer.evaluate(
                trained_deberta_model, test_loader, criterion,
                eval_type="Test", model_name='DeBERTa'
            )
            
            # Save results
            Utils.save_object(deberta_history, f"{self.deberta_model_dir}/training_history.pkl")
            Utils.save_object(deberta_test_results, f"{self.deberta_model_dir}/test_results.pkl")
            
            # Save model config
            config_deberta = {
                'embedding_dim': deberta_embedding_dim, 'tfidf_dim': tfidf_dim, 'feature_dim': feature_dim,
                'dropout': self.dropout, 'batch_size': self.batch_size, 'epochs': self.epochs,
                'learning_rate': self.lr, 'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'warmup_steps': warmup_steps,
                'best_val_acc': max(deberta_history['val_accuracy']) if deberta_history['val_accuracy'] else None,
                'test_acc': deberta_test_results['accuracy'],
                'test_loss': deberta_test_results['loss'],
                'timestamp': self.timestamp, 'base_dir': self.base_dir
            }
            Utils.save_object(config_deberta, f"{self.deberta_model_dir}/config.pkl")
            
            # Store results
            self.results['deberta']['model'] = trained_deberta_model
            self.results['deberta']['history'] = deberta_history
            self.results['deberta']['test_results'] = deberta_test_results
            
            print("DeBERTa training completed successfully")
            
        except Exception as e:
            print(f"Error during DeBERTa training: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_roberta_model(self):
        """Train and evaluate the RoBERTa model"""
        try:
            print("\n" + "="*50)
            print("Training RoBERTa Model")
            print("="*50)
            
            # Create dataloaders
            train_loader, val_loader, test_loader = self.data_manager.create_dataloaders(
                model_type='roberta', batch_size=self.batch_size
            )
            
            # Get model dimensions
            roberta_embedding_dim = self.data_manager.data['train']['roberta_embeddings'].shape[1]
            tfidf_dim = self.data_manager.data['train']['tfidf'].shape[1]
            feature_dim = self.data_manager.data['train']['features'].shape[1]
            
            print(f"RoBERTa Model dimensions - Embedding: {roberta_embedding_dim}, "
                  f"TF-IDF: {tfidf_dim}, Features: {feature_dim}")
            
            # Create model
            roberta_model = FakeNewsDetector(
                embedding_dim=roberta_embedding_dim, tfidf_dim=tfidf_dim,
                feature_dim=feature_dim, dropout=self.dropout
            )
            
            # Print model summary
            model_summary = roberta_model.summary()
            print("\nRoBERTa Model Summary:")
            for k, v in model_summary.items():
                print(f"{k}: {v:,}")
            
            # Calculate warmup steps
            num_training_steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
            total_training_steps = num_training_steps_per_epoch * self.epochs
            warmup_steps = int(total_training_steps * 0.1)
            print(f"RoBERTa - Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}")
            
            # Train model
            trained_roberta_model, roberta_history = self.trainer.train(
                roberta_model, train_loader, val_loader,
                epochs=self.epochs, lr=self.lr, warmup_steps=warmup_steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                model_dir=self.roberta_model_dir, model_name='roberta'
            )
            
            # Evaluate model on test set
            criterion = nn.BCEWithLogitsLoss()
            roberta_test_results = self.trainer.evaluate(
                trained_roberta_model, test_loader, criterion,
                eval_type="Test", model_name='RoBERTa'
            )
            
            # Save results
            Utils.save_object(roberta_history, f"{self.roberta_model_dir}/training_history.pkl")
            Utils.save_object(roberta_test_results, f"{self.roberta_model_dir}/test_results.pkl")
            
            # Save model config
            config_roberta = {
                'embedding_dim': roberta_embedding_dim, 'tfidf_dim': tfidf_dim, 'feature_dim': feature_dim,
                'dropout': self.dropout, 'batch_size': self.batch_size, 'epochs': self.epochs,
                'learning_rate': self.lr, 'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'warmup_steps': warmup_steps,
                'best_val_acc': max(roberta_history['val_accuracy']) if roberta_history['val_accuracy'] else None,
                'test_acc': roberta_test_results['accuracy'],
                'test_loss': roberta_test_results['loss'],
                'timestamp': self.timestamp, 'base_dir': self.base_dir
            }
            Utils.save_object(config_roberta, f"{self.roberta_model_dir}/config.pkl")
            
            # Store results
            self.results['roberta']['model'] = trained_roberta_model
            self.results['roberta']['history'] = roberta_history
            self.results['roberta']['test_results'] = roberta_test_results
            
            print("RoBERTa training completed successfully")
            
        except Exception as e:
            print(f"Error during RoBERTa training: {e}")
            import traceback
            traceback.print_exc()
    
    def _compare_models(self):
        """Compare the performance of both models"""
        try:
            if (self.results['deberta']['test_results'] is not None and 
                self.results['roberta']['test_results'] is not None):
                
                print("\n" + "="*50)
                print("Model Comparison")
                print("="*50)
                
                # Prepare comparison data
                results_dict = {
                    'DeBERTa': {
                        'accuracy': self.results['deberta']['test_results']['accuracy'],
                        'precision': self.results['deberta']['test_results']['precision'],
                        'recall': self.results['deberta']['test_results']['recall'],
                        'f1': self.results['deberta']['test_results']['f1'],
                    },
                    'RoBERTa': {
                        'accuracy': self.results['roberta']['test_results']['accuracy'],
                        'precision': self.results['roberta']['test_results']['precision'],
                        'recall': self.results['roberta']['test_results']['recall'],
                        'f1': self.results['roberta']['test_results']['f1'],
                    }
                }
                
                # Display comparison
                comparison_df = Utils.plot_metrics_comparison(
                    results_dict, 
                    self.run_dir,
                    display=self.display_plots
                )
                print("\nModel Performance Comparison:")
                print(comparison_df.to_string(index=False))
                
                # Full comparison dictionary including both models
                full_comparison = {
                    'timestamp': self.timestamp,
                    'deberta': {
                        'metrics': self.results['deberta']['test_results'],
                        'history': self.results['deberta']['history']
                    },
                    'roberta': {
                        'metrics': self.results['roberta']['test_results'],
                        'history': self.results['roberta']['history']
                    }
                }
                
                # Save full comparison
                Utils.save_object(full_comparison, f"{self.run_dir}/full_comparison.pkl")
                
        except Exception as e:
            print(f"Error during model comparison: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_results(self):
        """Create and save visualizations of the results"""
        try:
            print("\n" + "="*50)
            print("Generating Visualizations")
            print("="*50)
            
            # Plot training history for DeBERTa
            if self.results['deberta']['history']:
                Utils.plot_training_history(
                    self.results['deberta']['history'], 
                    'DeBERTa', 
                    self.deberta_model_dir,
                    display=self.display_plots
                )
                print("DeBERTa training history plots saved")
            
            # Plot training history for RoBERTa
            if self.results['roberta']['history']:
                Utils.plot_training_history(
                    self.results['roberta']['history'], 
                    'RoBERTa', 
                    self.roberta_model_dir,
                    display=self.display_plots
                )
                print("RoBERTa training history plots saved")
            
            # Plot confusion matrices
            if self.results['deberta']['test_results']:
                Utils.plot_confusion_matrix(
                    self.results['deberta']['test_results']['labels'],
                    self.results['deberta']['test_results']['preds'],
                    'DeBERTa',
                    self.deberta_model_dir,
                    display=self.display_plots
                )
                print("DeBERTa confusion matrix saved")
            
            if self.results['roberta']['test_results']:
                Utils.plot_confusion_matrix(
                    self.results['roberta']['test_results']['labels'],
                    self.results['roberta']['test_results']['preds'],
                    'RoBERTa',
                    self.roberta_model_dir,
                    display=self.display_plots
                )
                print("RoBERTa confusion matrix saved")
            
            # ROC Curve comparison
            if (self.results['deberta']['test_results'] and self.results['roberta']['test_results']):
                self._plot_roc_curves(
                    {
                        'DeBERTa': {
                            'labels': self.results['deberta']['test_results']['labels'],
                            'probs': self.results['deberta']['test_results']['probs']
                        },
                        'RoBERTa': {
                            'labels': self.results['roberta']['test_results']['labels'],
                            'probs': self.results['roberta']['test_results']['probs']
                        }
                    },
                    self.run_dir,
                    display=self.display_plots
                )
                print("ROC curve comparison saved")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_roc_curves(self, models_data, save_dir, display=True):
        """Plot ROC curves for model comparison"""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        for model_name, data in models_data.items():
            labels = data['labels']
            probs = data['probs']
            
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/roc_curve_comparison.png", dpi=300, bbox_inches='tight')
        
        if display:
            plt.show()
        else:
            plt.close()

# def main(base_dir='/kaggle/working', batch_size=64, epochs=20, lr=1e-4, 
#          gradient_accumulation_steps=1, dropout=0.3, display_plots=True):
#     """Main function to run the training pipeline"""
#     pipeline = FakeNewsTrainingPipeline(
#         base_dir=base_dir,
#         batch_size=batch_size,
#         epochs=epochs,
#         lr=lr,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         dropout=dropout,
#         display_plots=display_plots
#     )
    
#     results = pipeline.run()
    
#     # Print final summary
#     print("\n" + "="*60)
#     print("Training Pipeline Completed")
#     print("="*60)
    
#     # Print summary of both models
#     if results['deberta']['model'] and results['roberta']['model']:
#         print("\nModel Architecture Comparison:")
#         deberta_summary = results['deberta']['model'].summary()
#         roberta_summary = results['roberta']['model'].summary()
        
#         summary_df = pd.DataFrame({
#             'Parameter': list(deberta_summary.keys()),
#             'DeBERTa': [f"{v:,}" for v in deberta_summary.values()],
#             'RoBERTa': [f"{v:,}" for v in roberta_summary.values()]
#         })
#         print(summary_df.to_string(index=False))
    
#     # Print final metrics
#     if results['deberta']['test_results'] and results['roberta']['test_results']:
#         metrics_df = pd.DataFrame({
#             'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
#             'DeBERTa': [
#                 results['deberta']['test_results']['accuracy'],
#                 results['deberta']['test_results']['precision'],
#                 results['deberta']['test_results']['recall'],
#                 results['deberta']['test_results']['f1']
#             ],
#             'RoBERTa': [
#                 results['roberta']['test_results']['accuracy'],
#                 results['roberta']['test_results']['precision'],
#                 results['roberta']['test_results']['recall'],
#                 results['roberta']['test_results']['f1']
#             ]
#         })
        
#         print("\nFinal Test Metrics:")
#         print(metrics_df.to_string(index=False))
    
#     print(f"\nAll results and visualizations are saved in: {pipeline.run_dir}")
#     return results

# if __name__ == "__main__":
#     main(display_plots=True)  # Set to True to display plots in Kaggle notebook