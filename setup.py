import subprocess

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()


class CustomInstall(install):
    def run(self):
        subprocess.check_call(
            [
                "pip",
                "install",
                "--no-deps",
                "markushgrapher @ git+ssh://git@github.ibm.com/LUM/MarkushGrapher-IBM.git",
            ]
        )
        install.run(self)


class CustomDevelop(develop):
    def run(self):
        subprocess.check_call(
            [
                "pip",
                "install",
                "--no-deps",
                "markushgrapher @ git+ssh://git@github.ibm.com/LUM/MarkushGrapher-IBM.git",
            ]
        )
        super().run()


setuptools.setup(
    name="markushgenerator",
    version="1.0.0",
    author="Lucas Morin",
    author_email="lum@zurich.ibm.com",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        "svgpathtools",
        "lxml",
        "cairosvg",
        "datasets",
        "rdkit",
        "scikit-learn",
        "ipykernel",
        "matplotlib",
        "Pillow==9.5.0",
        "ipywidgets",
        "torch",
        "accelerate",
        "transformers",
        "SmilesPE",
        "numpy==1.24.4",
        "protobuf",
        "sentencepiece",
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
    cmdclass={
        "install": CustomInstall,
        "develop": CustomDevelop,
    },
    python_requires=">=3.9",
)
