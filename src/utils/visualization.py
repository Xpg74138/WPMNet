import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union


class ResultsVisualizer:
    """Visualization tool for model prediction results"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize visualization tool
        
        Args:
            class_names: List of class names for classification tasks
        """
        self.class_names = class_names or ["Class1", "Class2", "Class3"]
        self.cmap = plt.cm.viridis  # Use viridis colormap
    
    def visualize_classification_results(self, 
                                         preds: torch.Tensor, 
                                         targets: torch.Tensor,
                                         epoch: int = 0) -> Figure:
        """
        Visualize classification results
        
        Args:
            preds: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            epoch: Current training epoch
            
        Returns:
            matplotlib figure object
        """
        # Ensure data is on CPU
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # Get predicted classes
        if preds.shape[1] > 1:  # If probability distribution
            pred_classes = np.argmax(preds, axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int)  # Binary classification
            
        # Create confusion matrix
        cm = confusion_matrix(targets, pred_classes)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        ax1.set_title(f'Confusion Matrix (Epoch {epoch})')
        
        # 2. Accuracy per class
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        ax2.bar(self.class_names, accuracy_per_class, color=plt.cm.tab10.colors[:len(self.class_names)])
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy per Class')
        ax2.set_ylim(0, 1.0)
        
        for i, acc in enumerate(accuracy_per_class):
            ax2.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
            
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    def visualize_regression_results(self, 
                                    preds: torch.Tensor, 
                                    targets: torch.Tensor,
                                    epoch: int = 0) -> Figure:
        """
        Visualize regression results
        
        Args:
            preds: Model regression predictions (batch_size,)
            targets: Ground truth values (batch_size,)
            epoch: Current training epoch
            
        Returns:
            matplotlib figure object
        """
        # Ensure data is on CPU
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # Flatten arrays to ensure correct dimensions
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0))
        
        # 1. True values vs predicted values scatter plot
        ax1.scatter(targets, preds, alpha=0.6, color='blue')
        
        # Add ideal prediction line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax1.set_xlabel('Ground Truth')
        ax1.set_ylabel('Prediction')
        ax1.set_title(f'Ground Truth vs Prediction (Epoch {epoch})')
        
        # 2. Residual distribution
        residuals = preds - targets
        ax2.hist(residuals, bins=20, alpha=0.7, color='green')
        ax2.axvline(0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual (Prediction - Ground Truth)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        
        # Add residual mean and std information
        textstr = f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.close(fig)
        return fig
        
    def visualize_feature_maps(self, 
                            model: torch.nn.Module, 
                            images: torch.Tensor, 
                            target_layer: str = None) -> Figure:
        """
        Use basic feature visualization method, more robust
        
        Args:
            model: Trained model
            images: Input image tensor (batch_size, channels, height, width)
            target_layer: Target layer name, if None will automatically select last conv layer
            
        Returns:
            matplotlib figure object
        """
        try:
            device = images.device
            model = model.to(device)
            model.eval()
            
            # Take first image from batch
            img = images[0:1]
            
            # Hook function to store feature maps
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    # Special handling for output as dictionary
                    if isinstance(output, dict):
                        if 'regression' in output:
                            activation[name] = output['regression'].detach()
                    else:
                        activation[name] = output.detach()
                return hook
            
            # If target_layer not provided, try to find a suitable layer
            if target_layer is None:
                # Search for all conv layers
                conv_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        conv_layers.append(name)
                if conv_layers:
                    target_layer = conv_layers[-1]  # Use last conv layer
                else:
                    raise ValueError("No convolutional layers found, please specify target_layer manually")
            
            # Register hook to get feature map
            try:
                # Try to access submodules using dot notation
                parts = target_layer.split('.')
                current_module = model
                for part in parts:
                    if part.isdigit():  # If numeric index
                        current_module = current_module[int(part)]
                    else:
                        current_module = getattr(current_module, part)
                
                hook_handle = current_module.register_forward_hook(get_activation(target_layer))
            except (AttributeError, IndexError) as e:
                # If direct access fails, try using named_modules
                for name, module in model.named_modules():
                    if name == target_layer:
                        hook_handle = module.register_forward_hook(get_activation(target_layer))
                        break
                else:
                    raise ValueError(f"Layer not found: {target_layer}")
            
            # Forward pass to get features
            with torch.no_grad():
                _ = model(img)
            
            # Remove hook
            hook_handle.remove()
            
            # Get feature map
            feature_map = activation[target_layer]
            
            # If feature map is 4D [batch, channel, height, width]
            if feature_map.dim() == 4:
                # Average over all channels to get heatmap
                feature_map = feature_map.mean(1).squeeze(0)
            else:
                # For non-standard feature map shapes, special handling may be needed
                raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Display original image
            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            # If channels > 3, only take RGB or convert to single channel grayscale
            if img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
            
            # Normalize for display
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            ax1.imshow(img_np)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display activation map
            feature_map_np = feature_map.cpu().numpy()
            im = ax2.imshow(feature_map_np, cmap='jet')
            ax2.set_title(f'Feature Map for Layer "{target_layer}"')
            ax2.axis('off')
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.close(fig)
            return fig
            
        except Exception as e:
            # Create a figure with error information
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Feature visualization failed: {str(e)}\n\nTarget layer: {target_layer}", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            print(f"Feature visualization failed: {str(e)}")
            plt.close(fig)
            return fig

        
    def log_to_mlflow(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor], 
                     epoch: int,
                     model: torch.nn.Module = None,
                     sample_images: torch.Tensor = None,
                     target_layer: str = None):
        """
        Log visualization results to MLflow
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of ground truth values
            epoch: Current training epoch
            model: Model instance (for feature visualization)
            sample_images: Sample images (for feature visualization)
            target_layer: Target layer name (for feature visualization)
        """
        # Visualize classification results
        if 'classification' in predictions and 'classification' in targets:
            fig_cls = self.visualize_classification_results(
                predictions['classification'], 
                targets['classification'],
                epoch
            )
            mlflow.log_figure(fig_cls, f"visualization/classification_epoch_{epoch}.png")
            plt.close(fig_cls)
            
        # Visualize regression results
        if 'regression' in predictions and 'regression' in targets:
            fig_reg = self.visualize_regression_results(
                predictions['regression'],
                targets['regression'],
                epoch
            )
            mlflow.log_figure(fig_reg, f"visualization/regression_epoch_{epoch}.png")
            plt.close(fig_reg)
            
        # Feature visualization (if model and images are provided)
        if model is not None and sample_images is not None:
            try:
                fig_feat = self.visualize_feature_maps(model, sample_images, target_layer)
                mlflow.log_figure(fig_feat, f"visualization/feature_maps_epoch_{epoch}.png")
                plt.close(fig_feat)
            except Exception as e:
                print(f"Feature visualization failed: {str(e)}")
 