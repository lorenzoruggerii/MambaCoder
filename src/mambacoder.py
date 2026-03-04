"""Implementation of a Transcoder for Mamba model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_py.mambapy.mamba import MambaConfig, MambaForLM
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, Optional, List
from config import TranscoderConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Put in transcoderconfig paths to tokenizer and models_weights
device = "cuda" if torch.cuda.is_available() else "cpu"

class MambaCoder(nn.Module):

    def __init__(self, t_cfg: TranscoderConfig, model_cfg: MambaConfig):

        """
        Initializes the Mamba-Transcoder

        Args:
            t_cfg: config file for Transcoder
            model_cfg: config file for MambaForLM
        """

        super().__init__()
        self.cfg = t_cfg
        self.model_cfg = model_cfg

        # Load the model
        self.base_model = (
            MambaForLM.from_pretrained_safetensors(t_cfg.weight_path, model_cfg).to(device)
            if t_cfg.weight_path.endswith("safetensors") else 
            MambaForLM.from_pretrained(t_cfg.weight_path).to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(t_cfg.tokenizer_path)

        # Freeze the model parameters (this won't be trained)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get models dimensions
        self.num_layers = self.model_cfg.n_layers
        self.hidden_size = self.model_cfg.d_model
        self.output_shape = self.model_cfg.d_model * self.model_cfg.expand_factor

        # Define model name for saving
        self.model_name = f"TC_{self.num_layers}L_{self.hidden_size}H"

        # Create encoder for each layer
        self.encoders = nn.ModuleList([
            nn.Linear(self.hidden_size, t_cfg.num_features)
            for _ in range(self.num_layers)
        ]).to(device)

        # Define activation functions
        self.activation_functions = nn.ModuleList([
            t_cfg.activation_fn
            for _ in range(self.num_layers)
        ]).to(device)

        # Create decoders for each layer
        self.decoders = nn.ModuleList([
            nn.Linear(t_cfg.num_features, self.hidden_size) # going back from self.output_shape
            for _ in range(self.num_layers)
        ]).to(device)

        self._initialize_weights()

        # Initialize hooks and storage for activations
        self.hooks = []
        self.gate_acts = {}
        self.gate_inputs = {}

        # Feature importance tracking
        self.feature_importance = torch.zeros(self.num_layers, t_cfg.num_features).to(device)

    def _initialize_weights(self):

        for encoder_layer in range(self.num_layers):

            # Standard initialization
            std_encoder = 1.0 / np.sqrt(self.cfg.num_features)
            nn.init.uniform_(self.encoders[encoder_layer].weight, -std_encoder, std_encoder)
            nn.init.zeros_(self.encoders[encoder_layer].bias)

        for decoder_layer in range(self.num_layers):

            # Standard initializations
            std_decoder = 1.0 / np.sqrt(self.num_layers * self.hidden_size)
            nn.init.uniform_(self.decoders[decoder_layer].weight, -std_decoder, std_decoder)
            nn.init.zeros_(self.decoders[decoder_layer].bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model and cross-layer transcoder

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding

        Returns: 
            Dictionary containing:
                - 'last_hidden_state': The base model's output
                - 'feature_activations': Feature activations for each layer
                - 'reconstructed_activations': Reconstructed activations from channel path
        """

        # Clear all previous activations
        self.gate_acts.clear()
        self.gate_inputs.clear()

        # Forward pass through base model to collect inputs and outputs
        with torch.no_grad():
            logits = self.base_model(input_ids)
        # Now self.base_model cache is full of activations and inputs

        # Populate dictionaries with cache
        self.gate_inputs = {
            i: v['inputs'] for i, v in self.base_model.cache.items()
        }
        self.gate_acts = {
            i: v['out_block'] for i, v in self.base_model.cache.items() # here we are reconstructing x + Block(x)
        }

        # Process each layer's activations through the transcoder
        feature_activations = {} # save activations of Transcoder
        reconstructed_activations = {} # save reconstructed activations after TC

        for layer_idx in range(self.num_layers):
            if layer_idx in self.gate_acts:
                gate_input = self.gate_inputs[layer_idx]
                gate_output = self.gate_acts[layer_idx]

                # Normalize inputs and outputs
                gate_input_norm = F.layer_norm(gate_input, (gate_input.shape[-1],))
                gate_output_norm = F.layer_norm(gate_output, (gate_output.shape[-1],))

                # Encode to features
                features = self.encoders[layer_idx](gate_input_norm)
                features_activated = self.activation_functions[layer_idx](features)
                feature_activations[layer_idx] = features_activated

                # Reconstruct activations using decoder
                reconstructed = self.decoders[layer_idx](feature_activations[layer_idx])
                reconstructed_activations[layer_idx] = reconstructed

                # Update feature importance based on activation magnitude
                with torch.no_grad():
                    importance = torch.mean(torch.abs(features_activated), dim=(0, 1))
                    self.feature_importance[layer_idx] += importance
                
        return {
            'last_hidden_state': logits,
            'feature_activations': feature_activations,
            'reconstructed_activations': reconstructed_activations
        }

    def train_transcoder(self,
                         texts: List[str]
                         ) -> Dict[str, List[float]]:
        
        """
        Train the transcoder on a corpus of text.
        """

        # Set best loss to a very high number
        best_loss = 10_000

        # Set model to training mode
        self.train()

        all_params = (
            list(self.encoders.parameters())
            + list(self.decoders.parameters())
            + list(self.activation_functions.parameters())
        )

        # Create optimizer
        optimizer = torch.optim.Adam(
            all_params, lr=self.cfg.lr
        )

        # optimizer = torch.optim.SGD(
        #     params=all_params,
        #     lr=self.cfg.lr
        # )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.cfg.lr_scheduler_factor,
            patience=self.cfg.lr_scheduler_patience
        )

        # Training metrics
        metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'l0_metric': [],
            'learning_rate': []
        }

        # Tokenize all texts
        print("Tokenizing examples...")
        encoded_texts = [
            self.tokenizer.encode(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_length,
                padding="max_length"
                ).to(device)
            for text in texts
        ]
        print("End of tokenization!")

        # Training loop
        for epoch in range(self.cfg.num_epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_sparsity_loss = 0
            epoch_l0_metric = 0

            # Process in batches
            num_batches = (len(encoded_texts) + self.cfg.batch_size - 1) // self.cfg.batch_size

            # Define progress bar
            progress_bar = tqdm(range(num_batches), total=num_batches, colour="GREEN")

            for batch_idx in progress_bar:
                # Get batch
                start_idx = batch_idx * self.cfg.batch_size
                end_idx = min(start_idx + self.cfg.batch_size, len(encoded_texts))
                batch_texts = encoded_texts[start_idx:end_idx]

                # Take max_len for padding
                max_len = max(text.size(1) for text in batch_texts)
                padded_texts = []

                for text in batch_texts:
                    pad_len = max_len - text.size(1)
                    padded_text = F.pad(text, (0, pad_len), value=self.tokenizer.pad_token_id)
                    padded_texts.append(padded_text)

                input_ids = torch.cat(padded_texts, dim=0)

                # Clear gradients
                optimizer.zero_grad()

                self.gate_inputs.clear()
                self.gate_acts.clear()

                # Forward pass to collect activations
                with torch.no_grad():
                    self.base_model(input_ids)

                # Update dictionaries
                self.gate_inputs = {
                    i: v['inputs'] for i, v in self.base_model.cache.items() 
                }
                self.gate_acts = {
                    i: v['out_block'] for i, v in self.base_model.cache.items()
                }

                # Now model cache is full
                total_loss = 0
                reconstruction_loss = 0
                sparsity_loss = 0

                all_features = {} # Store features for reconstruction

                for layer_idx in range(self.num_layers):
                    if layer_idx in self.gate_inputs:
                        gate_input = self.gate_inputs[layer_idx]
                        gate_outputs = self.gate_acts[layer_idx]

                        # Normalize inputs and outputs
                        gate_input_norm = F.layer_norm(gate_input, gate_input.shape[-1:])
                        gate_output_norm = F.layer_norm(gate_outputs, gate_outputs.shape[-1:])

                        # Encode to features
                        features = self.encoders[layer_idx](gate_input_norm)

                        # Apply activation features
                        features_activated = self.activation_functions[layer_idx](features)
                        all_features[layer_idx] = features_activated

                        # Reconstruc gate outputs
                        reconstructed = self.decoders[layer_idx](features_activated)

                        # L0 metric (sparsity)
                        l0_metric = torch.mean((features_activated > 1e-6).float())

                        # Reconstruction loss
                        recon_loss = F.mse_loss(reconstructed, gate_output_norm)
                        reconstruction_loss += recon_loss

                        # L1 loss
                        l1_loss = 0 # because using topk, k is fixed

                        # Update total loss
                        total_loss += l1_loss + recon_loss

                # Backward pass and optimization
                total_loss.backward()

                # Clip gradients 
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                # Remove unreferenced memory
                del input_ids, batch_texts, padded_texts, gate_input, gate_outputs, features, features_activated, reconstructed

                # Free model's cache
                self.base_model.cache.clear()
                self.gate_inputs.clear()
                self.gate_acts.clear()

                # Empty CUDA cache
                torch.cuda.empty_cache()

                # Update metrics
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                if isinstance(sparsity_loss, torch.Tensor):
                    epoch_sparsity_loss += sparsity_loss.item()
                else:
                    epoch_sparsity_loss += sparsity_loss
                epoch_l0_metric += l0_metric.item()

                # Update progress bar
                progress_bar.set_description(f"Epoch: {epoch+1}, R: {recon_loss:4f}, T: {(total_loss.item()):4f}")
                progress_bar.update(1)

            # Record epoch metrics
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_sparsity_loss = epoch_sparsity_loss / num_batches
            avg_l0_metric = epoch_l0_metric / num_batches

            metrics['total_loss'].append(avg_total_loss)
            metrics['reconstruction_loss'].append(avg_recon_loss)
            metrics['sparsity_loss'].append(avg_sparsity_loss)
            metrics['l0_metric'].append(avg_l0_metric)
            metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # Step scheduler
            scheduler.step(avg_total_loss)

            print(f"Epoch {epoch+1}/{self.cfg.num_epochs}: "
                  f"Loss = {avg_total_loss:.4f}, "
                  f"Recon = {avg_recon_loss:.4f}, "
                  f"Sparsity = {avg_sparsity_loss:.4f}, "
                  f"L0 Metric = {avg_l0_metric:.4f}")
            
            # Save model if best one
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                self.save_model()

        return metrics
    

    def save_model(self):
        """Save the transcoder model."""
        torch.save({
            "encoders": self.encoders.state_dict(),
            "decoders": self.decoders.state_dict(),
            "feature_importance": self.feature_importance,
            "config": {
                "model_name": self.model_name,
                "num_features": self.cfg.num_features,
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size
            }
        }, self.cfg.save_path)

    @classmethod
    def load_model(cls, t_cfg, m_cfg):
        """Load a saved transcoder model"""
        checkpoint = torch.load(t_cfg.save_path, map_location=device)

        # Create model with saved config
        model = cls(
            t_cfg,
            m_cfg
        )

        # Load state dictionaries
        model.encoders.load_state_dict(checkpoint['encoders'])
        model.decoders.load_state_dict(checkpoint['decoders'])
        model.feature_importance = checkpoint['feature_importance']

        return model
    

if __name__ == "__main__":

    from mamba_py.mambapy.mamba import MambaForLM, MambaConfig
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, required=True, help="Dimensionality of the residual stream.")
    p.add_argument("--num_layers", type=int, required=True, help="Number of Mamba layers.")
    p.add_argument("--num_tokens", type=int, required=True, help="Tokenizer vocabulary size.")
    p.add_argument("--weight_path", type=str, required=True, help="Where Mamba weights are stored.")

    args = p.parse_args()

    m_cfg = MambaConfig(args.d_model, args.num_layers, args.num_tokens)
    t_cfg = TranscoderConfig(args.weight_path)

    tc = MambaCoder(t_cfg, m_cfg)

    # Load dataset
    dataset = load_dataset(
        'monology/pile-uncopyrighted',
        data_files='train/00.jsonl.zst',
        split='train'
    )
    dataset = dataset[:t_cfg.num_train_prompts]['text']

    # Start training
    tc.train_transcoder(dataset)




           



    


