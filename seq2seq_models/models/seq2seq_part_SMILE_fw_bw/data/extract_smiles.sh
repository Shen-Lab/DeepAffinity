#!/bin/bash

file_from=$1
file_to=$2
> ${file_to}
IFS=$'\n'
for line in $(cat $file_from);
do
  echo $line | tr "\t" "\n" | tail -n1 >> ${file_to}
done
