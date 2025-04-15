#!/bin/bash

for i in {1..3}
do
  python another_client.py &
done
for i in {1..3}
do
  python client.py &
done 

wait
