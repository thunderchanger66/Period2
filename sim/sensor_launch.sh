#!/bin/bash
~/study/sensor/build/lidar_node &
sleep 1
gz sim ~/study/sensor/sensor_tutorial.sdf
trap "kill 0" SIGINT
wait