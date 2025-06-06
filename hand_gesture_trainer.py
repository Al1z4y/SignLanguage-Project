#!/usr/bin/env python3
"""
Hand Gesture Recognition Model Trainer
Loads CSV dataset, preprocesses data, and trains TensorFlow model
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm

class HandGestureTrainer:
    def __init__(self, csv_file="psl_sign_language_keypoints.csv"):
        self.csv_file = csv_file
        self.df = None
        self.X = None
        self.y = None
        self.label_encoder = None
        self.scaler = None
        self.model = None
        
        # Model parameters
        self.test_size = 0.2
        self.validation_split = 0.2
        self.random_state = 42
        
    def load_and_preprocess_data(self):
        """Load CSV data and preprocess it"""
        print("Loading dataset...")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded dataset: {self.df.shape}")
        
        # Display basic info
        print(f"Columns: {len(self.df.columns)}")
        print(f"Alphabets: {self.df['alphabet'].nunique()}")
        print(f"Total samples: {len(self.df)}")
        
        # Remove unnecessary columns
        print("\nRemoving unnecessary columns...")
        columns_to_remove = ['sample_file']  # Remove filename column
        
        # Optional: Remove confidence columns (keep only x, y coordinates)
        remove_confidence = input("Remove confidence values? (y/n, default=y): ").lower().strip()
        if remove_confidence != 'n':
            confidence_cols = [col for col in self.df.columns if col.endswith('_confidence')]
            columns_to_remove.extend(confidence_cols)
            print(f"Removing {len(confidence_cols)} confidence columns")
        
        # Remove columns
        existing_cols_to_remove = [col for col in columns_to_remove if col in self.df.columns]
        if existing_cols_to_remove:
            self.df = self.df.drop(columns=existing_cols_to_remove)
            print(f"Removed columns: {existing_cols_to_remove}")
        
        print(f"Dataset shape after removal: {self.df.shape}")
        
        # Normalize pixel coordinates to 0-1 range (like MediaPipe)
        print("\nNormalizing pixel coordinates to 0-1 range...")
        
        # Separate coordinate columns
        x_columns = [col for col in self.df.columns if col.endswith('_x') and col != 'alphabet']
        y_columns = [col for col in self.df.columns if col.endswith('_y') and col != 'alphabet']
        
        print(f"Found {len(x_columns)} X coordinate columns")
        print(f"Found {len(y_columns)} Y coordinate columns")
        
        if len(x_columns) > 0 and len(y_columns) > 0:
            # Get original ranges for display
            x_min, x_max = self.df[x_columns].min().min(), self.df[x_columns].max().max()
            y_min, y_max = self.df[y_columns].min().min(), self.df[y_columns].max().max()
            print(f"Original coordinate ranges:")
            print(f"  X: {x_min:.1f} to {x_max:.1f}")
            print(f"  Y: {y_min:.1f} to {y_max:.1f}")
            
            # Normalize X coordinates (assuming typical video width range)
            # Find the actual range and normalize to 0-1
            for col in x_columns:
                col_min, col_max = self.df[col].min(), self.df[col].max()
                if col_max > col_min:  # Avoid division by zero
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
            
            # Normalize Y coordinates (assuming typical video height range)
            for col in y_columns:
                col_min, col_max = self.df[col].min(), self.df[col].max()
                if col_max > col_min:  # Avoid division by zero
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
            
            # Check normalized ranges
            x_min_norm, x_max_norm = self.df[x_columns].min().min(), self.df[x_columns].max().max()
            y_min_norm, y_max_norm = self.df[y_columns].min().min(), self.df[y_columns].max().max()
            print(f"Normalized coordinate ranges:")
            print(f"  X: {x_min_norm:.3f} to {x_max_norm:.3f}")
            print(f"  Y: {y_min_norm:.3f} to {y_max_norm:.3f}")
            print("✅ Coordinates normalized to match MediaPipe format (0-1 range)")
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        if missing_values > 0:
            print("Filling missing values with 0...")
            self.df = self.df.fillna(0)
        
        # Separate features and labels
        self.X = self.df.drop('alphabet', axis=1).values
        self.y = self.df['alphabet'].values
        
        print(f"Features shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        print("📊 Data now uses normalized coordinates (0-1) compatible with MediaPipe")
        
        return True
    
    def encode_labels(self):
        """Encode alphabet labels to numerical values"""
        print("\nEncoding labels...")
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print("Classes:", list(self.label_encoder.classes_)[:10], "..." if len(self.label_encoder.classes_) > 10 else "")
        
        return self.y_encoded
    
    def normalize_features(self):
        """Normalize feature values"""
        print("\nNormalizing features...")
        self.scaler = StandardScaler()
        self.X_normalized = self.scaler.fit_transform(self.X)
        
        print(f"Feature range after normalization:")
        print(f"  Min: {self.X_normalized.min():.4f}")
        print(f"  Max: {self.X_normalized.max():.4f}")
        print(f"  Mean: {self.X_normalized.mean():.4f}")
        print(f"  Std: {self.X_normalized.std():.4f}")
        
        return self.X_normalized
    
    def split_data(self):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={self.test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_normalized, self.y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y_encoded
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return True
    
    def create_model(self):
        """Create neural network model"""
        print("\nCreating model...")
        
        input_dim = self.X_train.shape[1]
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Input dimension: {input_dim}")
        print(f"Output classes: {num_classes}")
        
        # Create model architecture
        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # Dense layers with dropout
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the model"""
        print(f"\nTraining model for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.0001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_hand_gesture_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes, target_names=class_names))
        
        # Top-k accuracy
        for k in [1, 3, 5]:
            if len(class_names) >= k:
                top_k_acc = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                    self.y_test, y_pred, k=k
                ).numpy().mean()
                print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
        
        return test_accuracy, y_pred_classes
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Class distribution
        class_counts = pd.Series(self.y).value_counts()
        axes[1, 1].bar(range(len(class_counts)), class_counts.values)
        axes[1, 1].set_title('Class Distribution')
        axes[1, 1].set_xlabel('Alphabet Index')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_preprocessing(self):
        """Save model and preprocessing objects"""
        print("\nSaving model and preprocessing objects...")
        
        # Save model
        if self.model is not None:
            self.model.save('hand_gesture_model.h5')
            print("Model saved: hand_gesture_model.h5")
        
        # Save label encoder
        if self.label_encoder is not None:
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print("Label encoder saved: label_encoder.pkl")
        
        # Save scaler
        if self.scaler is not None:
            with open('feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print("Feature scaler saved: feature_scaler.pkl")
        
        # Save feature names
        feature_names = [col for col in self.df.columns if col != 'alphabet']
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        print("Feature names saved: feature_names.pkl")
    
    def create_confusion_matrix(self, y_true, y_pred):
        """Create and save confusion matrix"""
        print("\nCreating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def full_training_pipeline(self, epochs=100, batch_size=32):
        """Run the complete training pipeline"""
        print("Hand Gesture Recognition - Full Training Pipeline")
        print("=" * 55)
        
        try:
            # Load and preprocess data
            self.load_and_preprocess_data()
            
            # Encode labels
            self.encode_labels()
            
            # Normalize features
            self.normalize_features()
            
            # Split data
            self.split_data()
            
            # Create model
            self.create_model()
            
            # Train model
            history = self.train_model(epochs, batch_size)
            
            # Evaluate model
            test_accuracy, y_pred_classes = self.evaluate_model()
            
            # Plot results
            self.plot_training_history(history)
            
            # Create confusion matrix
            self.create_confusion_matrix(self.y_test, y_pred_classes)
            
            # Save everything
            self.save_model_and_preprocessing()
            
            print(f"\n✅ Training completed successfully!")
            print(f"Final test accuracy: {test_accuracy:.4f}")
            print(f"Model saved as: hand_gesture_model.h5")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("Hand Gesture Recognition Trainer")
    print("=" * 40)
    
    # Get CSV file path
    csv_file = input("Enter CSV file path (default: psl_sign_language_keypoints.csv): ").strip()
    if not csv_file:
        csv_file = "psl_sign_language_keypoints.csv"
    
    # Training parameters
    epochs = input("Enter number of epochs (default: 100): ").strip()
    epochs = int(epochs) if epochs else 100
    
    batch_size = input("Enter batch size (default: 32): ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    print(f"\nTraining parameters:")
    print(f"  CSV file: {csv_file}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Confirm
    confirm = input("\nStart training? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = HandGestureTrainer(csv_file)
    
    # Run training
    success = trainer.full_training_pipeline(epochs, batch_size)
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
    else:
        print("\n💥 Training pipeline failed!")

if __name__ == "__main__":
    main() 