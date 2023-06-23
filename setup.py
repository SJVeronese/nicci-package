import setuptools

setuptools.setup(
    name="nicci",
    version="0.13",
    url="",
    author="Simone Veronese",
    author_email="veronese@astron.nl",
    description="A collection of functions for the analysis of astronomical data cubes and images",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=['astropy','numpy','matplotlib','pandas','pvextractor','scipy','tqdm'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
