#!/bin/bash

cpu_cores=(0 1 2 3 4 5)
dir=/sys/devices/system/cpu

governor=$1

if [ "$governor" = "default" ]; then
	i=0
	while [ $i -lt 6 ]; do
		is_online=`cat $dir/cpu${cpu_cores[$i]}/online`
		if [ $is_online -eq 1 ]; then
			echo schedutil > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_governor;

			echo 2035200 > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_max_freq;
			echo 345600 > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_min_freq;
		fi
		i=`expr $i + 1`
	done
elif [ "$governor" = "userspace" ]; then
	i=0
	while [ $i -lt 6 ]; do
		is_online=`cat $dir/cpu${cpu_cores[$i]}/online`
		if [ $is_online == 1 ]; then
			echo userspace > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_governor;
		fi
		i=`expr $i + 1`
	done
	
	freq=$2
	i=0
	while [ $i -lt 6 ]; do
		is_online=`cat $dir/cpu${cpu_cores[$i]}/online`
		if [ $is_online -eq 1 ]; then
			cpu_cur_min_freq=`cat $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_min_freq`
			
			if [ $cpu_cur_min_freq -lt $freq ]; then
				echo $freq > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_max_freq;
				echo $freq > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_min_freq;
			elif [ $cpu_cur_min_freq -gt $freq ]; then
				echo $freq > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_min_freq;
				echo $freq > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_max_freq;
			else
				echo $freq > $dir/cpu${cpu_cores[$i]}/cpufreq/scaling_max_freq;
			fi
		fi

		i=`expr $i + 1`
	done
fi
