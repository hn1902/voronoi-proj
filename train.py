"""
Training Pipeline for Voronoi AlphaZero RL
Implements replay buffer, training loop, and iterative improvement.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from nn_model import VoronoiNet, create_model, load_trained_model
from state_encoder import StateEncoder
from self_play import SelfPlayEngine, TrainingExample, generate_training_data


class ReplayBuffer:
    """
    Stores training examples for neural network training.
    Implements a fixed-size buffer with random sampling.
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, example: TrainingExample):
        """Add a single example."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(example)
        else:
            self.buffer[self.position] = example
            self.position = (self.position + 1) % self.max_size
    
    def add_all(self, examples: List[TrainingExample]):
        """Add multiple examples."""
        for ex in examples:
            self.add(ex)
    
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample a random batch."""
        if len(self.buffer) <= batch_size:
            return self.buffer
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'buffer': self.buffer,
            'position': self.position,
            'max_size': self.max_size
        }, filepath)
        print(f"Replay buffer saved: {len(self.buffer)} examples to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from file."""
        if not os.path.exists(filepath):
            print(f"No replay buffer found at {filepath}")
            return
        
        data = torch.load(filepath)
        self.buffer = data['buffer']
        self.position = data['position']
        self.max_size = data['max_size']
        print(f"Replay buffer loaded: {len(self.buffer)} examples from {filepath}")


class VoronoiDataset(Dataset):
    """PyTorch Dataset for training examples."""
    
    def __init__(self, examples: List[TrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.state_tensor, ex.policy_target, ex.value_target


class Trainer:
    """
    Main training loop for AlphaZero-style learning.
    """
    
    def __init__(self, model: VoronoiNet, learning_rate: float = 0.001,
                 l2_weight: float = 1e-4, device: str = None):
        """
        Args:
            model: Neural network to train
            learning_rate: Learning rate for optimizer
            l2_weight: L2 regularization weight
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        
        # Device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
        
        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Training history
        self.history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'win_rate': []
        }
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step on a batch.
        
        Args:
            batch: Tuple of (states, policy_targets, value_targets)
            
        Returns:
            Dict of loss values
        """
        states, policy_targets, value_targets = batch
        
        # Move to device
        states = states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        # Forward pass
        policy_logits, values = self.model(states)
        
        # Policy loss: cross-entropy between predicted and MCTS policy
        # policy_logits: (batch, num_edges), policy_targets: (batch, num_edges)
        policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1).mean()
        
        # Value loss: MSE between predicted and actual outcome
        value_loss = self.value_loss_fn(values.squeeze(), value_targets)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with training examples
            
        Returns:
            Average losses
        """
        self.model.train()
        
        total_losses = []
        policy_losses = []
        value_losses = []
        
        for batch in dataloader:
            losses = self.train_step(batch)
            
            total_losses.append(losses['total_loss'])
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])
        
        avg_losses = {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }
        
        return avg_losses
    
    def train_on_examples(self, examples: List[TrainingExample],
                        batch_size: int = 32, epochs: int = 10,
                        verbose: bool = True) -> Dict[str, float]:
        """
        Train on a list of examples.
        
        Args:
            examples: Training examples
            batch_size: Batch size
            epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Final average losses
        """
        dataset = VoronoiDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print(f"Training on {len(examples)} examples for {epochs} epochs...")
        
        for epoch in range(epochs):
            losses = self.train_epoch(dataloader)
            
            # Record history
            self.history['total_loss'].append(losses['total_loss'])
            self.history['policy_loss'].append(losses['policy_loss'])
            self.history['value_loss'].append(losses['value_loss'])
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: "
                      f"loss={losses['total_loss']:.4f} "
                      f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f})")
        
        return losses


class IterativeTrainer:
    """
    Iterative training: self-play -> train -> evaluate -> repeat.
    """
    
    def __init__(self, model: VoronoiNet, num_iterations: int = 100,
                 games_per_iteration: int = 25, num_simulations: int = 400,
                 epochs_per_iteration: int = 10, batch_size: int = 32,
                 checkpoint_dir: str = 'models'):
        """
        Args:
            model: Neural network
            num_iterations: Number of training iterations
            games_per_iteration: Self-play games per iteration
            num_simulations: MCTS simulations per move
            epochs_per_iteration: Training epochs per iteration
            batch_size: Training batch size
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.num_simulations = num_simulations
        self.epochs_per_iteration = epochs_per_iteration
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Components
        self.encoder = StateEncoder()
        self.trainer = Trainer(model)
        self.replay_buffer = ReplayBuffer(max_size=500000)
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.best_model_path = None
    
    def run(self, verbose: bool = True):
        """
        Run the full iterative training loop.
        """
        print("=" * 60)
        print("Starting AlphaZero Iterative Training")
        print("=" * 60)
        print(f"Iterations: {self.num_iterations}")
        print(f"Games per iteration: {self.games_per_iteration}")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Device: {self.trainer.device}")
        print("=" * 60)
        
        for iteration in range(1, self.num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.num_iterations}")
            print(f"{'='*60}")
            
            # Phase 1: Self-play
            print("\n[Phase 1] Self-play...")
            start_time = time.time()
            
            engine = SelfPlayEngine(
                neural_net=self.model,
                encoder=self.encoder,
                num_simulations=self.num_simulations
            )
            
            examples, stats = engine.generate_games(
                num_games=self.games_per_iteration,
                verbose=(iteration == 1)  # Verbose on first iteration
            )
            
            self.replay_buffer.add_all(examples)
            
            sp_time = time.time() - start_time
            print(f"Self-play complete: {len(examples)} examples generated in {sp_time:.1f}s")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            
            # Phase 2: Training
            print("\n[Phase 2] Training...")
            start_time = time.time()
            
            # Sample from replay buffer
            training_examples = self.replay_buffer.sample(min(50000, len(self.replay_buffer)))
            
            losses = self.trainer.train_on_examples(
                training_examples,
                batch_size=self.batch_size,
                epochs=self.epochs_per_iteration,
                verbose=True
            )
            
            train_time = time.time() - start_time
            print(f"Training complete in {train_time:.1f}s")
            print(f"  Loss: {losses['total_loss']:.4f}")
            
            # Phase 3: Save checkpoint
            print("\n[Phase 3] Saving checkpoint...")
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_iter_{iteration}.pth')
            self.model.save_model(
                checkpoint_path,
                optimizer_state=self.trainer.optimizer.state_dict()
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save replay buffer
            buffer_path = os.path.join(self.checkpoint_dir, 'replay_buffer.pth')
            self.replay_buffer.save(buffer_path)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)


def train_from_scratch(num_iterations: int = 50, games_per_iteration: int = 20,
                       num_simulations: int = 200, checkpoint_dir: str = 'models'):
    """
    Convenience function to train from scratch.
    
    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Games per iteration
        num_simulations: MCTS simulations per move
        checkpoint_dir: Where to save models
    """
    # Create model
    encoder = StateEncoder()
    model = create_model(input_size=encoder.get_feature_size())
    
    print(f"Created new model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = IterativeTrainer(
        model=model,
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        num_simulations=num_simulations,
        checkpoint_dir=checkpoint_dir
    )
    
    # Run training
    trainer.run(verbose=True)


# Need to import F for the training code
import torch.nn.functional as F

if __name__ == '__main__':
    # Example usage
    print("Voronoi AlphaZero Training Pipeline")
    print("Run train_from_scratch() to start training")
