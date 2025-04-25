#!/bin/bash
docker run -v /home/gaolingxiao/LiteLang:/root/LiteLang -it --rm --privileged --gpus=all --workdir /root/LiteLang/ --entrypoint /bin/bash swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
