#!/bin/bash


pop_sizes=("1024" "2048" "4096" "8192" "16384" "32768" "65526" "131072" "262144")

for pop_size in "${pop_sizes[@]}"; do
	echo "Population Size:" $pop_size
	time ./bin/performance_tests -p $pop_size
done
