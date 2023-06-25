from setuptools import setup, find_packages

setup(
    author="Bc. Jiří Kadlec",
    author_email="jiri.kadlec.st@vsb.cz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Vysoka Skola Banska - Technicka Univerzita Ostrava",
        "Topic :: Deep Learning :: CNN",
        "Programming Language :: Python :: 3.9.5",
    ],
    description="A series DL models (CNNs, RNNs) used for correct prediction on a chosen set of datasets.",
    install_requires=[
                        "numpy",
                        "matplotlib",
                        "tensorflow==2.10.0",
                        "scikit-learn",
                        "pandas"
                    ],
    include_package_data=True,
    keywords="dl, hu, rnn, cnn, ml, university, algorithm, vsb, tuo",
    long_description="",
    name="CNN_basic_package",
    packages=find_packages("src"),
    python_requires=">=3.9, <4",
    setup_requires=["setuptools"],
    version="0.0.1",
)