from distutils.core import setup
from setuptools import setup
from setuptools import find_packages



setup(name='delft',
      version='0.1',
      description='Complex Factoid Question Answering with a Free-Text Knowledge Graph',
      author='Chen Zhao, Chenyan Xiong, Xin Qian and Jordan Boyd-Graber',
      author_email='chenz@cs.umd.edu',
      packages=['delft'],
      url="delft.qanta.org",
      install_requires=[
          'nltk',
          'numpy',
          'scipy',
          'sklearn',
          'torch',
          'torchsummary',
          'dgl',
          'tqdm',
          'pytorch_transformers',
      ], 
      )