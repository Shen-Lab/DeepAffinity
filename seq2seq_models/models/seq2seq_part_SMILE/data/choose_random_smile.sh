#!/bin/bash

file_from=$1
num_smile=$2
cp $file_from temp
total_len=$(wc -l temp| awk '{print $1}')
IFS=$'\n'
i=0
while [ $i -lt $num_smile ]
do
  rand_num=$(shuf -i 1-$total_len -n 1)
  line=$(sed "${rand_num}q;d" temp)
  len=$(echo $line | wc -c  | awk '{print $1}') 
  if [ $len -lt 100 ];then
      echo $line
      sed -i "${rand_num}d" temp
      total_len=$((total_len - 1))
      i=$((i+1))
  fi
done
rm temp
