# Neuracl

Neuracl is a high-performance Spike Neural Network (SNN) simulation library written in Rust. It leverages GPU acceleration to provide efficient and scalable simulations of various SNN models, layers, topographies, synapse types, learning methods, and more.

## Features

- **GPU Acceleration**: Utilizes GPU power to accelerate simulations, supporting multiple GPU platforms.
- **Versatile Models**: Supports various types of models, layers, topographies, synapse types, learning methods, and neuron models.
- **Flexible Simulation**: Offers both propagating/updating neurons based on spike events and traditional simulating/synchronizing methods.
- **File Management**: Allows loading and saving different model states and configurations.
- **User-Friendly API**: Designed with an easy-to-use API without sacrificing control and customizability.

## Installation

Add `neuracl` to your `Cargo.toml` file:

```toml
[dependencies]
neuracl = "0.1.0"
```

## Usage

Here's a simple example of how to use Neuracl to create and run a simulation:

```rust
// Import the neuracl crate
use neuracl::*;

// Create a model, layers, neurons, etc.
// ...

// Run the simulation
// ...
```

For more detailed examples and tutorials, please refer to the [documentation](link-to-documentation).

## Documentation

Comprehensive documentation is available [here](link-to-documentation).

## Contributing

Contributions are welcome! Please read our [contributing guidelines](link-to-contributing-guidelines) for details on how to contribute to Neuracl.

## License

Neuracl is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to contributors, supporters, and everyone involved in the project.
