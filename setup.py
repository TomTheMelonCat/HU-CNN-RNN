from setuptools import setup, find_packages

setup(
    author="Bc. Jiří Kadlec",
    author_email="jiri.kadlec.st@vsb.cz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Vysoka Skola Banska - Technicka Univerzita Ostrava",
        "Topic :: Deep Learning :: RNN/CNN",
        "Programming Language :: Python :: 3.9.5",
    ],
    description="A series DL models (CNNs, RNNs) used for correct prediction on a chosen set of datasets.",
    install_requires=[
                        "numpy",
                        "matplotlib",
                        "tensorflow",
                        "scikit-learn",
                        "pandas"
                    ],
    include_package_data=True,
    keywords="dl, hu, rnn, cnn, ml, university, algorithm, vsb, tuo",
    long_description="",
    name="deep_learning",
    package_dir={"package": "src"},
    packages=find_packages("src"),
    package_data={"package": ["py.typed"]},
    python_requires=">=3.9, <4",
    setup_requires=["setuptools"],
    version="0.0.1",
)