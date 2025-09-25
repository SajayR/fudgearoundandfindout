#!/bin/bash

# Setup script for DinoV2 LoRA Testbed

set -e

echo "Setting up DinoV2 LoRA Testbed..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o "3\.[0-9]\+")
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python version: $python_version"

# Check if we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: Please run this script from the fisher-lora directory"
    exit 1
fi

# Create virtual environment (optional)
read -p "Create a virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Dependencies installed successfully"

# Check ImageNet dataset
echo "Checking ImageNet dataset..."
if [[ -d "/speedy/ImageNet/train" && -d "/speedy/ImageNet/val" ]]; then
    train_classes=$(ls /speedy/ImageNet/train | wc -l)
    val_classes=$(ls /speedy/ImageNet/val | wc -l)
    echo "✓ ImageNet dataset found:"
    echo "  - Train classes: $train_classes"
    echo "  - Val classes: $val_classes"
else
    echo "⚠️  ImageNet dataset not found at /speedy/ImageNet/"
    echo "   Please ensure the dataset is available at the expected location"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs checkpoints experiments evaluation_results
echo "✓ Directories created"

# Run quick test
echo "Running setup verification..."
python scripts/quick_test.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment (if created): source venv/bin/activate"
echo "2. Run a quick training test: python train.py --config default --override training.epochs=1"
echo "3. Run baseline experiments: bash scripts/run_baseline_experiment.sh"
echo ""
echo "For more information, see README.md"
