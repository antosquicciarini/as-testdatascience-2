from setuptools import find_packages, setup

setup(
    name='as-testdatascience-2',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description='Multivariate time series forecasting of home energy consumption using the Appliances Energy Prediction dataset (UCI). The goal is to predict future energy use without access to future values of regressors. Includes preprocessing, stationarity check, model training, and evaluation. Fully reproducible project structure.',
    author='Antonio Squicciarini',
    license='MIT',
)
