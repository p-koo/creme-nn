from setuptools import setup, find_packages

setup(
    name="creme-nn",
    # TODO: Consider using https://github.com/python-versioneer/python-versioneer to
    # get version information from git.
    version="0.2.5",
    description="An in silico perturbation framework to interpret large-scale genomic deep learning",
    # author='A. Random Developer',
    # author_email='author@example.com',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(),
    python_requires=">=3.6, <4",
    # Do not install tensorflow here, because might want to use tensorflow or
    # tensorflow-cpu.
    install_requires=[
        "pyranges",
        "numpy",
        "pandas",
        "kipoiseq",
        "pyfaidx",
        "logomaker",
        "matplotlib",
        "seaborn",
        "protobuf==3.20",
    ],
    extras_require={
        "dev": [
            "black",  # styler
            "flake8",  # linter
        ],
    },
)
