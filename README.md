
## Clemson COVID challenge

The Clemson COVID challenge was a summer virtual research and design opportunity for teams of faculty and undergraduates to work on problems related to the COVID-19 Pandemic as well as creating solutions for future pandemics. With partner university University of South Carolina and Prisma health, Teams had a little more than half a month to tackle a problem in the areas of communication, Education, Healthcare Technology, Policy/Economy/Logistics, or Society/Community.

Focusing on the area of Healthcare Technology, My mentors Dr. Dane Smith, Dr. Carl Ehrett, and I decided to work on building a privacy-centric, affordable, open-source fever detection solution. With a team of students and me at the helm, four weeks of hard work converged to a solution conveniently named the Tig**IR** which ticked many of the boxes we wanted while coming in at sub $500.


## Background on using thermal cameras for Fever Detection

For over 30 years, IR thermal cameras have been used to diagnose issues in many industries everything from healthcare applications to home cooling. This is because heat generated from sources emit a band of light that the human eye or any standard camera can perceive. However, when targeting said band, we can get the temperature of a point in space by the amount of light it emits. This is then mapped to a color map and creates images for diagnosing problems

In the case of faces, we can use thermal imaging to get the temperature of a subject's face which will emit more IR energy if they have a fever aka an elevated body temperature

Normally we could use this data with a calibrated sensor to get the exact temperature at certain spots, however this spot in which a person has to measure temperature is very specific and therefore sometimes hard to capture

## Component Selection
Like I said before, we wanted the Tig**IR** to be an affordable solution and using off the shelf parts. Luckily there were a few options for each which gave us some flexibility

### The Brain

Being able to run who machine learning models and process incoming images can be a resource intensive task. For an effective solution you have two options
1. Nvidia Jetson Nano
+ has Cuda cores for dedicated machine learning techniques
+ built in, very effective heat sink
+ has pcie slot for storage or wifi
+ Overclockable with a very capable arm processor
+ Expensive


2. Raspberry Pi 4 4GB model
+ Affordable
+ Build in Wifi and bluetooth
+ Wider community support
+ Neo-arm processor which is faster however dedicated GPU not as strong
+ micro-hdmi


### The Sensors
For our solution to work correctly, it requires two sensors one that sees the visible spectrum and the other 
that sees IR. Instead of listing all the possible options, let me give you the reasons for the products we selected:

1. LABISTS Raspberry Pi noIR Camera V2
* Sony mirrorless 8MP sensor capable of 1080p at 30fps
* IR filter removed for better low light performance
* Super easy to install with ribbon cable 
* Low cost of $27.99 USD

3. FLIR Radiometric Lepton V2.5 
* 80x60 IR solution for $250 USD
* FLIR is known for their quality products, reliability, and documentation

### Enclosure
The enclosure we selected for this project was selected based on its features and price. The ** Miuzei Case** includes a fan and 4 aluminum heat sinks for cooling. This case also includes a 15W power supply which covered that component. The IO on this enclosure is really easy to access.

### Misc
Some components that we had to purchase that are generic:
* MicroSD Card
* Breadboard Cables

### Prototype development
In order to have a test bed for developing code, I built a testbed to hold the sensors while we were finishing designing and prototyping the 3D printable enclosure.

## Setting up the Pi 

### Preparing the SD Card
Setting up a Raspberry Pi is quite simple these days. Using a computer with a microSD card slot or an external SD card reader, plug your microSD card into your computer and then allow the system to have writing permissions to said storage device (should be enabled by default). Next head over to [Raspberry OS Imager Guide](https://www.raspberrypi.org/documentation/installation/installing-images/README.md) where you can download the Pi Imager and install your preferred version of Raspbian. 

After the image is installed, create a txt file in the main directory of the microSD card named **SSH** to enable ssh forwarding. This allows you to connect to the Pi from a PC over a local network instead of having to find a mini-hdmi cable. Using **Windows Remote Desktop Protocol** you can even view the desktop in real time. This is where I did most of the development for this project. 

In hopes that the Raspberry Pi community would have it's own OS image with the tools necessary to perform machine learning tasks on the Pi, my search came up with nothing but images locked behind pay walls that costs hundreds of dollars. In search for a cheap alternative solution, we have created an Image that contains OpenCV, Numpy, Pandas, Pytorch, and dlib compiled from source to run on the neo-arm architecture sufficiently so that you don't have to spend hours on lengthy tutorials. You can download that [Here](blank)

### SSH into Pi
To get the ip of the raspberry pi on your network simply type in a windows or Linux terminal 
```
ping raspberrypi
```


### Installing the packages
There are quite a number of packages that are necessary for this project along with their requirements:
* OpenCV with dlib
* Numpy
* Pandas
* Pylepton
* Pytorch
* Picamera

Many of these were compiled from source using cmake instead of pip installing so that they could take advantage of the neo-arm architecture. Tutorials for these libraries are available.

### Enabling IO
Using the Raspberry Pi configuration tool, make sure to enable the use of the GPIO pins and CSI ribbon slot. Once enabled, shut down the Pi and plug in the normal camera to the ribbon slot and plug each GPIO pin to its respective position as shown below. 

To test to see if the normal camera is working type the following into a terminal which will generate at test image:

```
raspistill -o testshot.jpg
```
If the image is not generated check your connections and Pi configuration again. 




## The Code

```markdown
Syntax highlighted code block

# Header 1
This is a test
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
