from setuptools import setup, find_packages

setup(
    name='GRNgene',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'matplotlib',
        'pandas',
        'scipy',
        'openpyxl'
    ],
    extras_require={
        'dev': ['pytest'],
    },
    package_data={
        "GRNgene": ["ODESystems/41586_2011_BFnature10098_MOESM304_ESM.xls","GRN/41598_2021_3625_MOESM5_ESM.xlsx"],
    },
    author='Alexandre Tan-Lhernould',
    author_email='tlalex@hotmail.fr',
    description="Parametric Generator of Synthetic Gene Networks and Expression Data",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TLAlex/GRNgene',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
