#!/bin/bash

for i in {1..3}
do
  python another_client.py &
done
sleep 1
# for i in {1..3}
# do
#   python client.py &
# done

wait
