from setuptools import setup, find_packages

setup(
    name='VonML',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='Christian Jenei',
    author_email='christian.oliver.jenei@gmail.com',
    description='A tiny machine learning library.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
