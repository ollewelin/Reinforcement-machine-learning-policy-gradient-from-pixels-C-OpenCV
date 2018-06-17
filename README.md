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
the screen may look like this:

$ ./exe_main
Reinforcment Learning test of pixels data input from a simple ping/pong game
Construct a arcade game object
Number of hidden nodes to one frames = 40
Total number of hidden nodes fo all frames together = 4000
Number of output nodes alway equal to the number of frames on one episode = 100
Insert noise to weights. Please wait...
Noise to the weight pixel to hidden is inserted
Noise to the weight hidden to output node is inserted
Would you like to add noise on input image <Y>/<N> 
n
********** No image noise **********
Would want to use default settings <Y>/<N> 
y
********** Default settings **********
pix2hid_learning_rate = 0.400000
hid2out_learning_rate = 0.100000
Would you like to load stored weights, pix2hid_weight.dat and hid2out_weight.dat <Y>/<N> 
Example use (if you don't already trainied some good files) <N>
y
Start so load pix2hid_weight.dat Please wait... The file size is = 40000000 bytes
weights are loaded from pix2hid_weight.dat file
Start so load hid2out_weight.dat Please wait... The file size is = 16000 bytes
weights are loaded from hid2out_weight.dat file
..
..
  
(if you get 
Error while opening file pix2hid_weight.dat
you maybee foreget extract the pix2hid_weight.dat.tar.gz)

