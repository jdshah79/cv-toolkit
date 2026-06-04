from setuptools import setup, find_packages

setup(
    name="my_cv_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "ultralytics"
    ],
)
