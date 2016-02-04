cmake_minimum_required(VERSION 2.8)
project( CaptureImage )
find_package( OpenCV REQUIRED )
add_executable( CaptureImage otrack.cpp )
target_link_libraries( CaptureImage ${OpenCV_LIBS} )

