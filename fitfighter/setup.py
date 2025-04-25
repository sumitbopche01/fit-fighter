from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fitfighter",
    version="0.1.0",
    author="FitFighter Team",
    author_email="info@fitfighter.example.com",
    description="Exercise detection and fitness tracking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fitfighter/fitfighter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.10",
        "numpy>=1.19.0",
        "websockets>=10.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "fitfighter=fitFighter.src.main:main",
        ],
    },
)
