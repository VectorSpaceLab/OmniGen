from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='OmniGen',
    version='1.0.0',
    description='OmniGen',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='2906698981@qq.com',
    url='https://github.com/VectorSpaceLab/OmniGen',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.41.0',
        'datasets',
        'accelerate>=0.20.1',
    ],
)