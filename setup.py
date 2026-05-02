from setuptools import setup, find_packages

setup(
    name="cow-breed-detection",
    version="0.1.0",
    description="Deep learning project for cow breed detection and classification",
    author="AELISH",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "numpy>=1.24",
        "pandas>=1.5",
        "matplotlib>=3.7",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "pyyaml>=6.0",
    ],
)
