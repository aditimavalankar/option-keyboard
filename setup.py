from setuptools import setup, find_packages

setup(name='option_keyboard',
      version='0.1',
      description='PyTorch implementation of The Option Keyboard',
      author='Aditi Mavalankar',
      author_email='amavalan@eng.ucsd.edu',
      url='https://github.com/aditimavalankar/option-keyboard',
      install_requires=[
            'torch', 'gym', 'numpy', 'tensorboardX'
      ],
      packages=find_packages())
