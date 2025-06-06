import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="Pyriod",
    version="0.3.2",
    author="Keaton Bell",
    author_email="keatonbell@utexas.edu",
    description="Basic period detection and fitting routines for astronomical time series.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keatonb/Pyriod",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)

