from setuptools import setup, find_packages

setup(
    name="robomimic",
    packages=[
        package for package in find_packages() if package.startswith("robomimic")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "psutil",
        "tqdm",
        "termcolor",
        "tensorboard",
        "tensorboardX",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "egl_probe>=1.0.1"
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="robomimic: A Modular Framework for Robot Learning from Demonstration",
    author="Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang",
    url="https://github.com/ARISE-Initiative/robomimic",
    author_email="amandlek@cs.stanford.edu",
    version="0.3.0",
    long_description='',
    long_description_content_type='text/markdown'
)
