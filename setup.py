import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyriod",
    version="0.0.1",
    author="Keaton Bell",
    author_email="keatonbell@utexas.edu",
    description="Sinusoid fitting for the astronomical time domain.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keatonb/Pyriod",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
