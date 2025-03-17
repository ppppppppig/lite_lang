#!/bin/bash

for i in {1..5}
do
  python client.py &
done

wait
