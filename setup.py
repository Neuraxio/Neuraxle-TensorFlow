"""
Neuraxle Tensorflow Utility classes
=========================================
Neuraxle utility classes for tensorflow.

..
    Copyright 2019, Neuraxio Inc.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from setuptools import setup, find_packages

from neuraxle_tensorflow import __version__ as _VERSION

with open('README.md') as _f:
    _README = _f.read()

setup(
    name='neuraxle_tensorflow',
    version=_VERSION,
    description='TensorFlow steps, savers, and utilities for Neuraxle. Neuraxle is a Machine Learning (ML) library for building neat pipelines, providing the right '
                'abstractions to both ease research, development, and deployment of your ML applications.',
    long_description_content_type='text/markdown',
    long_description=_README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Telecommunications Industry",
        'License :: OSI Approved :: Apache Software License',
        "Natural Language :: English",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Topic :: Adaptive Technologies",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Assemblers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Topic :: Software Development :: Object Brokering,
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: System",
        # Topic :: System :: Clustering,
        # Topic :: System :: Distributed Computing,
        # Topic :: System :: Networking,
        # Topic :: System :: Systems Administration,
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Filters",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        "Typing :: Typed"
    ],
    url='https://github.com/Neuraxio/Neuraxle-Tensorflow',
    download_url='https://github.com/Neuraxio/Neuraxle-Tensorflow/tarball/{}'.format(
        _VERSION),
    author='Neuraxio Inc.',
    author_email='guillaume.chevalier@neuraxio.com',
    packages=find_packages(include=['neuraxle_tensorflow*']),
    test_suite="testing",
    setup_requires=["pytest-runner"],
    install_requires=[
        'neuraxle>=0.3.1'
    ],
    tests_require=["pytest", "pytest-cov"],
    include_package_data=True,
    license='Apache 2.0',
    keywords='pipeline pipelines data science machine learning deep learning'
)

print("""
Thank you for installing
   _   _                               __
  | \ | |                             |  |
  |  \| | ___  _   _  _ __  ___ __  __ | |  ___
  | . ` |/ _ \| | | || ' _||__ \\\\ \/ / | | / _ \\
  | |\  || __|| |_| | | |  / _ | >  <  | | | __|
  |_| \_|\___| \__,_||___| \_,_|/_/\_\ |__|\___|
     ==> TensorFlow Edition.
____________________________________________________________________
 Thank you for installing neuraxle-tensorflow.
  
 Learn more:
 - https://www.neuraxle.org/stable/index.html
 Contribute:
 - https://gitter.im/Neuraxle/community
 Open issue:
 - https://github.com/Neuraxio/Neuraxle
 Ask questions:
 - https://stackoverflow.com/questions/tagged/neuraxle
____________________________________________________________________
""")
