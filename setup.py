import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="markushgenerator",
    version="1.0.0",
    author="Lucas Morin",
    author_email="lum@zurich.ibm.com",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    package_data={"markushgenerator": ["data/*.json", "data/vocabulary/*.json"]},
    install_requires=[
        "svgpathtools",
        "lxml",
        "cairosvg",
        "datasets",
        "rdkit",
        "scikit-learn",
        "ipykernel",
        "matplotlib",
        "Pillow>=10.3.0",
        "ipywidgets",
        "torch",
        "accelerate",
        "transformers",
        "SmilesPE",
        "numpy==1.24.4",
        "protobuf",
        "sentencepiece"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    extras_require={
        "grapher": [
            "markushgrapher @ git+https://git@github.com/DS4SD/MarkushGrapher.git",
        ],
    },
    python_requires=">=3.9",
)
