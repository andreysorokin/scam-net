from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='scam-net-rewintous',
    packages=['scam'],
    version='0.0.1',
    license='	apache-2.0',
    description='Score Weighted Class Activation Mapping. A tool for convolutional neural network activation analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Andrei Sorokin',
    author_email='Andrey.I.Sorokin@gmail.com',
    url='https://github.com/andreysorokin/scam-net',
    download_url='https://github.com/andreysorokin/scam-net/archive/0.0.1.tar.gz',
    keywords=['CNN', 'neural', 'heatmap'],
    install_requires=[
        'keras',
        'numpy',
        'scikit-image',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
