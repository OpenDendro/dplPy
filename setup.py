from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setuptools.setup(
    name="dplpy-opendendro",
    version="0.0.1",
    author="Tyson Lee Swetnam",
    author_email="tswetnam@arizona.edu",
    description="The dplPy package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opendendro/dplpy",
    project_urls={
        "Funding": "https://nsf.gov/awardsearch/showAward?AWD_ID=2054516",
        "Bug Tracker": "https://github.com/opendendro/dplpy/issues",
        "Changelog": "https://github.com/opendendro/dplpy/blob/main/CHANGELOG.md"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)