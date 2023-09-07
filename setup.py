from setuptools import setup

install_requires = [
    "numba",
    "numpy",
    "scikit-learn",
]

setup(
    name="numbadecisiontrees",
    version="0.0.1",
    description="novel 'numba' based recreation of scikit-learn's decision tree algorithm",
    url="https://github.com/pr38/numbadecisiontrees",
    author="Pouya Rezvanipour",
    author_email="pouyar3@yahoo.com",
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
    ],
    install_requires=install_requires,
    packages=["numbadecisiontrees"],
)
