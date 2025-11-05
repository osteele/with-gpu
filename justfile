# Build the project
build:
    cargo build --release

# Install to ~/.cargo/bin
install:
    cargo install --path .

# Run cargo check
check:
    cargo check

# Run cargo clippy
lint:
    cargo clippy -- -D warnings

# Format code
format:
    cargo fmt

# Check formatting
format-check:
    cargo fmt -- --check

# Run all checks (format, lint, check)
all-checks: format-check lint check

# Clean build artifacts
clean:
    cargo clean

# Show GPU status (requires nvidia-smi)
status:
    cargo run -- --status

# Test with a simple command
test-echo:
    cargo run -- echo "Testing with-gpu wrapper"

# Test manual GPU selection
test-manual GPU="0":
    cargo run -- --gpu {{GPU}} echo "Using GPU {{GPU}}"
