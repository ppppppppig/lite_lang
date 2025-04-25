#!/bin/bash

for i in {1..50}
do
  python another_client.py &
done
for i in {1..25}
do
  python client.py &
done 

wait
