import setuptools
from setuptools_rust import Binding, RustExtension

setuptools.setup(
    name="great-model",
    version="1.0.0",
    author="Metodi Nikolov",
    author_email="metodi.nikolov@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    rust_extensions=[RustExtension("great_model.rust_great_model", binding=Binding.PyO3)],
    zip_safe=False,
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)