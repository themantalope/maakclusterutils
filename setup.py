from setuptools import setup

setup(
    author="Matthew Antalek",
    name="maakclusterutils",
    packages=["maakclusterutils"],
    license="MIT",
    author_email="matthew.antalek@northwestern.edu",
    version="0.1dev",
    install_requires = ["tqdm", "pathos"]
)