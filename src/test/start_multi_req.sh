#!/bin/bash

for i in {1..1}
do
  python another_client.py &
done
for i in {1..1}
do
  python client.py &
done 

wait
