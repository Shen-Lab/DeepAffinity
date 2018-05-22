#!/bin/bash

file_from=$1
num_smile=1000
cp $file_from temp_here
for i in $(seq 1 10);
do
   num=$((i*50000))
   head -n${num} temp_here > mytemp_here
   num=$(((10-i)*50000))
   tail -n${num} $file_from > temp_here
   ./choose_random_smile.sh mytemp_here 120
done
rm mytemp_here
rm temp_here
