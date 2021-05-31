from setuptools import setup

setup(
   name='branch2learn',
   version='1.0',
   description='Ablating a Graph Neural Network for Branching in Mixed Integer Linear Programming',
   author='Lars Sandberg',
   author_email='larslsa@stud.ntnu.no',
   packages=['branch2learn'],
   install_requires=['numpy', 'pytorch'],
)