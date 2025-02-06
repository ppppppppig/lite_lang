#!/bin/bash
docker run -v /home/gaolingxiao/LiteLang:/root/LiteLang -it --rm --privileged --gpus=all --workdir /root/LiteLang/src/kernel --entrypoint /bin/bash af91b83f4e9e