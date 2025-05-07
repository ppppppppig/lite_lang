#!/bin/bash
docker run -v /home/gaolingxiao/LiteLang:/root/LiteLang -it --rm --privileged --gpus=all --workdir /root/LiteLang/src/kernel --entrypoint /bin/bash f30da6d7a30d
