from setuptools import setup

setup(
    name="sfat",
    version="0.1.0",
    description="Sparse Frequency-Aware Transformer for Spiking Neural Networks",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "spikingjelly>=0.0.0.0.14",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "tonic>=1.0.0",
        "opencv-python>=4.7.0",
    ],
    extras_require={
        "log": ["wandb"],
    },
)
