# Reinforcement-machine-learning-policy-gradient-from-pixels-C-OpenCV
Pong game catch circles but avoid rectangles 

Youtube demo at:
https://www.youtube.com/watch?v=fO5CFOeZen8&t=43s

dependencies: OpenCV
Here is a greate install guide for OpenCV: 
http://milq.github.io/install-opencv-ubuntu-debian/

Tested on Linux Ubuntu OpenCV 3.1

Compile

$make

Run

$./exe_main

If you want to test my trained weights ~100000 training episodes
combine the splited 8M files pix2hid_weight.parta* with this command

cat pix2hid_weight.parta* > pix2hid_weight.dat.tar.gz

then extract the file:
pix2hid_weight.dat.tar.gz

start program and enter 
N
Y
Y
on the 3 comming question 
  
(if you get 
Error while opening file pix2hid_weight.dat
you maybee foreget extract the pix2hid_weight.dat.tar.gz)

