import setuptools

setuptools.setup(
    name="great-model",
    version="0.0.1",
    author="Metodi Nikolov",
    author_email="metodi.nikolov@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)