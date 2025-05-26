from setuptools import setup, find_packages

setup(
    name="litelang",
    version="0.8.0",
    description="一个轻量级的大语言模型推理部署框架",
    author="高凌霄",
    author_email="gaolingxiao@sensetime.com",
    packages=find_packages(),
    package_data={},
    install_requires=[
        "pyzmq",
        "uvloop",
        "transformers",
        "einops",
        "packaging",
        "rpyc",
        "ninja",
        "safetensors",
        "triton==2.2.0",
        "fastapi",
        "uvicorn",
        "sortedcontainers"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "litelang=litelang.apis.test_server:main",
        ],
    },
)
