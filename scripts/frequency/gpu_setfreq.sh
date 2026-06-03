#!/bin/bash

dir=/sys/devices/17000000.gp10b/devfreq/17000000.gp10b
default_max=1300500000
default_min=114750000
freq=$1

gpu_curminfreq=`cat /sys/devices/gpu.0/devfreq/17000000.gp10b/min_freq | tr -d '\r\n'`

if [ "$freq" == "default" ]; then
	echo $default_max > $dir/max_freq;
	echo $default_min > $dir/min_freq;
else
	if [ $gpu_curminfreq -lt $freq ]; then
		echo $freq > $dir/max_freq;
		echo $freq > $dir/min_freq;
	elif [ $gpu_curminfreq -gt $freq ]; then
		echo $freq > $dir/min_freq;
		echo $freq > $dir/max_freq;
	else 
		echo $freq > $dir/max_freq;
	fi
fi

echo "Current GPU freq: `cat /sys/devices/gpu.0/devfreq/17000000.gp10b/cur_freq | tr -d '\r\n'`"