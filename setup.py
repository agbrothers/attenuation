from setuptools import find_packages, setup

setup(
    name="attenuation",
    version="0.0.0",
    author="Greyson Brothers",
    author_email="greysonbrothers@gmail.com",
    description="Implementation and experiments for the Attenuation paper",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/agbrothers/attenuation",
    packages=[package for package in find_packages() if package.startswith("attenuation")],
    zip_safe=False,
    install_requires=[
        "matplotlib",
        "torch",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/agbrothers/attenuation/issues",
    },
    classifiers=["Programming Language :: Python :: 3",],
    python_requires=">=3.6",
)
