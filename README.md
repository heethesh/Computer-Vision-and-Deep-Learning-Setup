# Computer Vision  and Deep Learning Setup

Tutorial on how to setup your system with a NVIDIA GPU and to install Deep Learning Frameworks like TensorFlow, Darknet for YOLO, Theano and Keras, OpenCV and NVIDIA drivers, CUDA and cuDNN libraries on Ubuntu 16, 17 and 18.


## 1. Install Prerequisites
Before installing anything, let us first update the information about the packages stored on the computer and upgrade the already installed packages to their latest versions.

	sudo apt-get update
	sudo apt-get upgrade

Next, we will install some basic packages which we might need during the installation process as well in future.

	sudo apt-get install -y build-essential cmake gfortran git pkg-config 

**NOTE: The following instructions are only for Ubuntu 17 and 18.04. Skip to the next section if you have Ubuntu 16.04**
	
The defualt *gcc* vesrion on Ubuntu 17 and 18.04 is *gcc-7*. However, when we build OpenCV from source with CUDA support, it requires *gcc-5*. 

	sudo apt-get install gcc-5 g++-5
	
Verify the *gcc* version:

	gcc --version
	
You may stil see version 7 detected. We have to set higher priority for *gcc-5* as follows (assuming your *gcc* installation is located at */usr/bin/gcc-5*, and *gcc-7*'s priority is less than 60.

	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 60
	
Now, to fix the CXX and CC environment variable systemwide, you need to put the lines in your .bashrc file:

	echo 'export CXX=/usr/bin/g++-5.4' >> ~/.bashrc
	echo 'export CC=/usr/bin/gcc-5.4' >> ~/.bashrc
	source ~/.bashrc


## 2. Install NVIDIA Driver for your GPU
The NVIDIA drivers will be automatically detected by Ubuntu in *Software and Updates* under *Additional drivers*. Select the driver for your GPU and click apply changes and reboot your system. *You may also select and apply Intel Microcode drivers in this window.*

*At the time of writing this document, the latest stable driver version is 396*.

Run the following command to check whether the driver has installed successfully by running NVIDIA’s System Management Interface (*nvidia-smi*). It is a tool used for monitoring the state of the GPU.

	nvidia-smi
	
#### NOTE: You will have to disable SecureBoot in your BIOS and change boot mode from UEFI to Legacy, else you might face the login loop issue. An alternate way to go about this issue with UEFI systems is to enroll the MOK keys as explained [here](https://gist.github.com/Rambou/c6769caee19b0b9915d8342b86c3ef72).

## 3. Install CUDA
CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by NVIDIA which utilizes the parallel computing capabilities of the GPUs. In order to use the graphics card, we need to have CUDA drivers installed on our system.

Download the CUDA driver from the official nvidia website [official nvidia website](https://developer.nvidia.com/cuda-92-download-archive). We recommend you download the *deb (local)* version from Installer type as shown in the screenshot below.

*At the time of writing this document, the latest stable version is CUDA 9.2*.

**NOTE: The following instructions will also work for Ubuntu 17 and 18.04.**

![](https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/blob/master/images/img1.png) 

After downloading the file, go to the folder where you have downloaded the file and run the following commands from the terminal to install the CUDA drivers. Please make sure that the filename used in the command below is the same as the downloaded file.

	sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
	sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
	sudo apt-get update
	sudo apt-get install -y cuda-9

Now, you have to install the CUDA performance update patch available from the same webpage where you downloaded the CUDA Base Installer.
	
	sudo dpkg -i cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64
	sudo apt-get update

Next, update the paths for CUDA library and executables.

	echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64:/usr/local/cuda-9.2/extras/CUPTI/lib64"' >> ~/.bashrc
	echo 'export CUDA_HOME=/usr/local/cuda-9.2' >> ~/.bashrc
	echo 'export PATH="/usr/local/cuda-9.2/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc
	
You can verify the installation of CUDA version by running:

	nvcc -V
	
#### NOTE: The procedure for installing CUDA 10.0 will be similiar as above
	
## 4. Install cuDNN
CUDA Deep Neural Network (cuDNN) is a library used for further optimizing neural network computations. It is written using the CUDA API.

Go to official cuDNN website [official cuDNN website](https://developer.nvidia.com/cudnn)  and fill out the form for downloading the cuDNN library. 

*At the time of writing this document, the latest stable version is CuDNN 7.4*.

**Make sure you download the correct CuDNN version which matches with you CUDA version.**

![](https://github.com/heethesh/Computer-Vision-and-Deep-Learning-Setup/blob/master/images/img2.png) 

Now, go to the folder where you have downloaded the “.tgz” file and from the command line execute the following.

	tar xvf cudnn-9.2-linux-x64-v7.4.tgz
	sudo cp -P cuda/lib64/* /usr/local/cuda-9.2/lib64/
	sudo cp cuda/include/* /usr/local/cuda-9.2/include/
	sudo chmod a+r /usr/local/cuda-9.2/include/cudnn.h
	
To check installation of cuDNN, run this in your terminal:
	
	dpkg -l | grep cudnn
	
## 5. Python and Other Dependencies
Now, we install Tensorflow, Keras and Theano along with other standard Python ML libraries like numpy, scipy, sklearn etc.

Install dependencies of deep learning frameworks:

	sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libopencv-dev

Next, we install Python 2 and 3 along with other important packages like boost, lmdb, glog, blas etc.

	sudo apt-get install -y --no-install-recommends libboost-all-dev doxygen
	sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev libblas-dev 
	sudo apt-get install -y libatlas-base-dev libopenblas-dev libgphoto2-dev libeigen3-dev libhdf5-dev 
	 
	sudo apt-get install -y python-dev python-pip python-nose python-numpy python-scipy python-wheel
	sudo apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy python3-wheel
	
**NOTE: If you want to use Python3, replace the following pip commands with pip3.**

Before we use pip, make sure you have the latest version of pip.

	sudo pip install --upgrade pip

## 6. Install Deep Learning Frameworks
Now, we can install all the required python packages for deep learning frameworks:

	sudo pip install numpy scipy matplotlib scikit-image scikit-learn ipython protobuf jupyter
	
Upgrade numpy to the latest version:

	sudo pip install --upgrade numpy
	 
### Building TensorFlow from Source

**NOTE: If you already have a built &ast;.whl file, skip to next sub-section to install TensorFlow directly using PIP**

Now we will download the TensorFlow repository from GitHub in the */home* folder. Checkout to the latest version of TensorFlow (`r1.5` is used here).

	cd ~
	git clone https://github.com/tensorflow/tensorflow.git
	cd tensorflow
	git checkout r1.5
	
You must also install libcupti which for Cuda Toolkit >= 8.0 you do via:
	
	sudo apt-get install cuda-command-line-tools-9-2 
	
For Cuda Toolkit <= 7.5, you install libcupti-dev by invoking the following command:
	
	sudo apt-get install libcupti-dev 
	
Next we need to install Bazel
	
	sudo apt-get install openjdk-8-jdk
	sudo apt-get install curl
	sudo apt-get install zip zlib1g-dev unzip
	
	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	sudo apt-get update && sudo apt-get install bazel
	sudo apt-get upgrade bazel
	
	wget https://github.com/bazelbuild/bazel/releases/download/0.11.0/bazel-0.11.0-installer-linux-x86_64.sh
	chmod +x bazel-0.11.0-installer-linux-x86_64.sh
	./bazel-0.11.0-installer-linux-x86_64.sh --user
	
	export PATH="$PATH:$HOME/bin"
	source ~/.bashrc

To verify installation of Bazel run:
	
	bazel version

Now install brew on your system:

	sudo apt-get install linuxbrew-wrapper
	brew doctor
	brew install coreutils
	
The root of the *tensorflow* folder contains a bash script named configure. This script asks you to identify the pathname of all relevant TensorFlow dependencies and specify other build configuration options such as compiler flags. You must run this script prior to creating the pip package and installing TensorFlow.

	cd ~/tensorflow
	./configure

#### NOTE: Enter the your correct CUDA and CuDNN version below. CUDA 9.2 and CuDNN 7.4 is used here

>Select Python 2.7, no to all additional packages, gcc as compiler (GCC 5.4).
>
>For CUDA, enter 9.2
>
>For cuDNN, enter 7.4
>
>Enter your GPU Compute Capability (Eg: 3.0 or 6.1). Find yout GPU Compute Capability from [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
>
>Use nvcc as the CUDA compiler.

Finally, build the pip package:
	
	bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package 

The build might take upto an hour. If it fails to build, you must clean your build using the following command and configure the build once again.

	bazel clean --expunge
	./configure

The bazel build command builds a script named build_pip_package. Running this script as follows will build a .whl file within the /tmp/tensorflow_pkg directory:

	bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

### Installing TensorFlow using PIP

Once the build is complete, invoke pip install to install that pip package. The filename of the .whl file depends on your platform. Use tab completion to find your package.

	sudo pip install /tmp/tensorflow_pkg/tensorflow <TAB>
	
If you get an error saying package is not supported for the current platform, run pip as pip2 for Python 2.7:

	sudo pip2 install /tmp/tensorflow_pkg/tensorflow <TAB> (*.whl)

You can make a backup of this .whl file.

	cp /tmp/tensorflow_pkg/tensorflow <TAB> (*.whl) <BACKUP_LOCATION>
	
An alternate way to install TensorFlow using pip without building is as follows:

	sudo pip install tensorflow-gpu

Verify that TensorFlow is using the GPU for computation by running the following python script.

**NOTE: Running a script from the */tensorflow* root directory might show some errors. Change to any other directory and run the following python script.**

	import tensorflow as tf
	with tf.device('/gpu:0'):
	    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
	    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	    c = tf.matmul(a, b)
	
	with tf.Session() as sess:
	    print (sess.run(c))

Here,

- "*/cpu:0*": The CPU of your machine.
- "*/gpu:0*": The GPU of your machine, if you have one.

If you have a gpu and can use it, you will see the result. Otherwise you will see an error with a long stacktrace. 

Now, install keras for TensorFlow and Theano
 
	sudo pip install keras
	sudo pip install Theano	

Check installation of frameworks:

	python
	>>> import numpy
	>>> numpy.__version__
	>>> import tensorflow
	>>> tensorflow.__version__
	>>> import keras
	>>> Using TensorFlow backend.
	>>> keras.__version__
	>>> import theano
	>>> theano.__version__

## 7. Install OpenCV and Contrib Modules
First we will install the dependencies:

	sudo apt-get remove -y x264 libx264-dev
	sudo apt-get install -y checkinstall yasm
	sudo apt-get install -y libjpeg8-dev libjasper-dev libpng12-dev
	
	sudo apt-get install -y libtiff5-dev
	sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
	 
	sudo apt-get install -y libxine2-dev libv4l-dev
	sudo apt-get install -y libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
	sudo apt-get install -y libqt4-dev libgtk2.0-dev libtbb-dev
	sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
	sudo apt-get install -y libvorbis-dev libxvidcore-dev
	sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
	sudo apt-get install -y x264 v4l-utils

#### NOTE: Checkout to the latest version of OpenCV. 3.4.4 is used here

Download OpenCV 3.4.4:

	git clone https://github.com/opencv/opencv.git
	cd opencv
	git checkout 3.4.4
	cd ..

Download OpenCV-contrib 3.4.4:

	git clone https://github.com/opencv/opencv_contrib.git
	cd opencv_contrib
	git checkout 3.4.4
	cd ..
	
#### NOTE: Keep the build folder in the same location as it may be required in future to upgrade or uninstall OpenCV

Configure and generate the MakeFile in */opencv/build* folder (make sure to specify paths to downloaded OpenCV-contrib modules correctly):

	cd opencv
	mkdir build
	cd build
	
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	      -D CMAKE_INSTALL_PREFIX=/usr/local \
	      -D INSTALL_C_EXAMPLES=ON \
	      -D INSTALL_PYTHON_EXAMPLES=ON \
	      -D WITH_TBB=ON \
	      -D WITH_V4L=ON \
	      -D WITH_QT=ON \
	      -D WITH_OPENGL=ON \
	      -D WITH_CUDA=ON \
	      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	      -D BUILD_EXAMPLES=ON ..

#### NOTE: If you are using Python3, you must add the following flag as well

	      -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
	      
#### NOTE: If you are using Ubuntu 17 or 18.04, you must add the following flags as well

	      -D CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-5 \
	      -D CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-5 \
	
Compile and install:

	make -j$(nproc)
	sudo make install
	sudo ldconfig
	
Check installation of OpenCV:

	python
	>>> import cv2
	>>> cv2.__version__
	
Retain the build folder in the same location. This will be required if you want to uninstall OpenCV or upgrade in the future or else the uninstall process might become very tedious.

#### NOTE: If you get any errors with `import cv2`, make sure the `PYTHONPATH` points to the location of `cv2.so` file correctly in your `~/.bashrc` file as follows

	export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
	export PYTHONPATH=/usr/local/lib/python3.5/site-packages:$PYTHONPATH

To uninstall OpenCV:

	cd /opencv/build
	sudo make uninstall
	
## 8. Install Darknet for YOLO

First clone the Darknet git repository.

	git clone https://github.com/pjreddie/darknet.git

Now, to compile Darknet with CUDA, CuDNN and OpenCV support, open the `Makefile` from the `darknet` folder and make the changes as following in the beginning of this file.

	GPU=1
	CUDNN=1
	OPENCV=1
	
Once done, just run make from the darknet folder.

	cd darknet
	make

Refer [here](https://pjreddie.com/darknet/yolo/) for more details on running YOLO and training the network.
