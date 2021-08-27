# wasp_summer_school_21
Group 2

# Steps to run
- Open a terminal (#1)
- Run ```cd opendlv-perception-helloworld-cpp``` to enter opendlv-perception-helloworld-cpp directory.
- Run ```docker build -t myapp .``` to build docker image.
- Open another terminal (#2), run ```docker-compose -f h264-replay-viewer.yml up``` to start data replay.
- In terminal #1, run ```docker run --rm -ti --init --net=host --ipc=host -v /tmp:/tmp -e DISPLAY=$DISPLAY myapp --cid=111 --name=img.argb --width=1280 --height=720 --steer=0.5 --throttle=0.1 --accgain=0.5 --verbose``` to run the docker image.
- Try different values for ```steer```, ```throttle```, and ```accgain```, to tune performance of the application.
