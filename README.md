<h1 align="center">
  <br>
  <a href="https://github.com/N4D1K-lgtm/Neuracl"><img src="https://github.com/N4D1K-lgtm/Neuracl/blob/9d10b5c23c1d8027b28ece3960a814de0fbe84b4/assets/Neuracl.png" alt="Neuracl" width="200"></a>
  <br>
  Neuracl
  <br>
</h1>


<h4 align="center">A <a href="https://www.rust-lang.org/" target="_blank">Rust</a> powered Spike Neural Network (SNN) simulation library inspired by Tensorflow.</h4>


<p align="center">
  <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://badge.fury.io/js/electron-markdownify.svg"
         alt="Gitter">
  </a>
  <a href="https://gitter.im/amitmerchant1990/electron-markdownify"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
  <a href="https://saythanks.io/to/bullredeyes@gmail.com">
      <img src="https://img.shields.io/badge/SayThanks.io-%E2%98%BC-1EAEDB.svg">
  </a>
  <a href="https://www.paypal.me/AmitMerchant">
    <img src="https://img.shields.io/badge/$-donate-ff69b4.svg?maxAge=2592000&amp;style=flat">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#usage">How To Use</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

## Overview
Neuracl is an experimental and highly parallelized library for simulating and experimenting with Spike Neural Networks (SNNs), designed to bridge the gap between computational models and biological systems like the human brain. With a firm focus on speed, versatility, and precision, Neuracl provides researchers, engineers, and enthusiasts a powerful tool to create, train, visualize, fine-tune, and test various SNN architectures. 

## Key Features

- **Highly Optimized**: Written in Rust, Neuracl harnesses the power of CUDA, allowing GPU acceleration, ensuring optimal performance without sacrificing quality.
- **Versatility**: Support for various models, layers, topographies, synapse types, learning methods, and neuron models. Experiment without limitations!
- **Model Management**: Conveniently save, load, and manage models, preserving your progress and enabling smooth transitions between experiments.
- **User-Friendly API**: Easily create complex simulations with an intuitive and customizable API that never compromises on control.
- **Type Safety**: Built on the strong and reliable Rust toolchain, ensuring robust type safety, error handling, and code quality.
- **Visualization Tools**: Analyze and understand your networks through comprehensive visualization tools, helping to demystify complex patterns and behaviors.

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
