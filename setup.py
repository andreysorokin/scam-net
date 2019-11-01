from distutils.core import setup
setup(
  name = 'scam-net',
  packages = ['scam-net'],
  version = '0.0.1',
  license='	apache-2.0',
  description = 'Score Weighted Class Activation Mapping. A tool for convolutional neural network activation analysis',
  author = 'Andrei Sorokin',
  author_email = 'Andrey.I.Sorokin@gmail.com',
  url = 'https://github.com/andreysorokin/scam-net',
  download_url = 'https://github.com/andreysorokin/scam-net/archive/0.0.1.tar.gz',
  keywords = ['CNN', 'neural', 'heatmap'],
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