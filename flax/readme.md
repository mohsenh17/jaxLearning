# Model Checkpointing with PRNG Key Restoration

This repository demonstrates how to implement model checkpointing and restore a model's state, including handling the restoration of PRNG keys used in dropout layers, in a JAX-based neural network.

## Overview

The script showcases the following key functionalities:
1. **Model Definition**: Defines an MLP model using the `nnx` (Neural Network Extensions) library.
2. **Training & Evaluation**: Implements training and evaluation steps for the model.
3. **Checkpointing**: Saves and restores the model state using `orbax`, ensuring that all parameters and random number generator (RNG) states (including for dropout layers) are correctly restored.
4. **PRNG Key Restoration**: Demonstrates how to properly handle and restore PRNG keys associated with layers like dropout during model restoration.
5. **State Comparison**: Compares the original and restored model states to ensure consistency after restoration.

## Dependencies

- `jax`
- `optax`
- `nnx`
- `orbax`
- `numpy`
- `torch`
- `flax`

## Key Components

- **Model Definition**: The `MLP` class defines a multi-layer perceptron with dropout and batch normalization layers.
- **Training Step**: The `train_step` function computes the loss and performs backpropagation with gradients.
- **Checkpointing**: The model’s state is saved and restored using the `PyTreeCheckpointer` from `orbax`. Special care is taken to restore the PRNG keys used in dropout layers.
- **State Restoration**: The restored model is checked for consistency with the original state using `jax.tree.map(np.testing.assert_array_equal)`.
- **Custom Metrics**: Accuracy, precision, recall, and f1_score can be used to meassure the predictions 

## Usage

1. **Training**: 
   - Define your dataset and use the `train_step` to train the model.
2. **Saving Checkpoint**:
   - Save the model state using `checkpointer.save()`.
3. **Restoring Checkpoint**:
   - Restore the model state from a checkpoint using `checkpointer.restore()`, ensuring the correct PRNG keys are restored for dropout layers.
4. **Reinitialize Model**:
   - After restoration, the model can be reinitialized with the restored state using `nnx.merge()`.
5. **Custom Metric**:
   - Use accuracy, precision, recall, and F1 score to evaluate your predictions.

## Example

```python
# Initialize model
model = MLP(10, 16, 32, 16, 1, rngs=nnx.Rngs(0))

# Save model state
checkpointer.save('/content/my-checkpoints/state', state)

# Restore model state
state_restored = checkpointer.restore('/content/my-checkpoints/state', abstract_state)

# Reinitialize model with restored state
newModel = nnx.merge(graphdef, state_restored)
