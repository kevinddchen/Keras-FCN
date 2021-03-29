#!/bin/bash
input="val.txt"
while IFS= read -r line
do
  mv $line.jpg val
done < "$input"
