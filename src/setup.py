import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
        "Bug Tracker": "https://github.com/opendendro/dplpy/issues",
        "Changelog": "https://github.com/opendendro/dplpy/blob/main/CHANGELOG.md"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8.6",
)