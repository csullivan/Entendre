#!/bin/bash


#num_networks=("65536" "131072" "262144")
#num_networks=("1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072" "262144")
num_networks=("1024" "2048" "4096" "8192" "16384" "32768" "65536")


for num_network in "${num_networks[@]}"; do
	echo "Number of networks to evaluate:" $num_network
	time ./bin/performance_tests -n $num_network
done
