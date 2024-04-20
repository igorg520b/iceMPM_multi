# iceMPM_multi
Multi-GPU implementation of MPM for modeling of ice (2D version)

> apt update && apt install libeigen3-dev libspdlog-dev libcxxopts-dev rapidjson-dev libhdf5-dev cmake mc -y

For GUI version, enable the option BUILD_GUI_VERSION and install VTK9:

> apt install libvtk9-dev

Video can be rendered from .png images in "screenshot" folder with the following command:

> ffmpeg -y -r 60 -f image2 -start_number 1 -i "%05d.png" -vframes 2400 -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf 15  -pix_fmt yuv420p "result.mp4"

To run from command line, the path to Qt 5.15.2 library must be set:

> export LD_LIBRARY_PATH="/home/s2/Qt/5.15.2/gcc_64/lib"

![Screenshot of the GUI version](/screenshot.png)
