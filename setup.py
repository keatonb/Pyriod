import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="Pyriod",
    version="0.0.1",
    author="Keaton Bell",
    author_email="keatonbell@utexas.edu",
    description="Sinusoid fitting for the astronomical time domain.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keatonb/Pyriod",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)
