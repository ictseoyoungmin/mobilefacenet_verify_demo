from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'my_script=my_package.demo:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)