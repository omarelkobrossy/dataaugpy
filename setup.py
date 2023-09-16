from setuptools import setup, find_packages

setup(
    name='dataaug-python',
    version='0.1',
    packages=find_packages(),
    install_requires=["numpy", "opencv-python"],  # Add dependencies if needed
)
