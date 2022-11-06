from setuptools import setup

# python3 setup.py sdist bdist_wheel

setup(
    name='matrix_free_nnls',
    version='1.0',
    packages=['matrix_free_nnls'],
    url='https://github.com/FabianKP/matrix_free_nnls',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'typing',
    ],
    author='FabianKP',
    author_email='',
    description='Implementation of accelerated projected gradient descent for '
                'matrix-free non-negative least-squares problems.'
)