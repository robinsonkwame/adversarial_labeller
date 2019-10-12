#!/usr/bin/env python

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

with open('requirements_dev.txt') as requirements_file:
    test_requirements = [
        req for req in requirements_file.readlines()
        if 'git+' not in req
    ] + requirements

setup(
    name='Adversarial Validator',
    version='0.1.0',
    description='Sklearn compatiable model selection tool for validation models that scores train instances that look the most like test instances.',
    author="Kwame Porter Robinson",
    author_email='kwamepr@umich.edu',
    url='https://github.com/robinsonkwame/adversarial_validator',
    packages=find_packages(include=['adversarial_validator*']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    keywords='model selection, validation, data drift',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
)