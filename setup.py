from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyAPRiL",
    version="1.7.6",
    author="Tamas Peto",
    author_email="petotax@gmail.com",
    description="A python based passive radar library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petotamas/APRiL",
    packages=find_packages(),
    classifiers=[        
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)  ",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)