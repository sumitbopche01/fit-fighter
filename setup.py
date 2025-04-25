from setuptools import setup, find_packages

setup(
    name="fitfighter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
        "matplotlib>=3.7.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.1",
        "websockets>=11.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "tf": [
            "tensorflow>=2.13.0",
        ],
    },
    author="FitFighter Team",
    description="A motion detection system for exercise tracking",
    keywords="fitness, exercise, pose estimation",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
