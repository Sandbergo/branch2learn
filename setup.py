from setuptools import setup

setup(
   name='branch2learn',
   version='0.1',
   description='Learning to Branch in Mixed Integer Linear Programming',
   author='Lars Sandberg',
   author_email='larslsa@stud.ntnu.no',
   packages=['branch2learn'],
   install_requires=['numpy', 'pytorch'],
)