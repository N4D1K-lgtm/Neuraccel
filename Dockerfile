# Use NVIDIA's CUDA image as the base
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Install dependencies required for Rust installation
RUN apt-get update && apt-get install -y curl

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Verify Rust installation
RUN rustc --version
RUN cargo --version

# Install LLVM/Clang (if needed)
RUN apt-get install -y llvm clang

# Set up the working directory
WORKDIR /usr/src/app

# Copy the Cargo manifest files
COPY Cargo.toml Cargo.lock ./
RUN cat Cargo.toml

# Cache dependencies
RUN cargo fetch

# Copy the source code
COPY src ./src

# Build the project
RUN cargo build --release

# Command to run the application
CMD ["cargo", "run", "--release"]