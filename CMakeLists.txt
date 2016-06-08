cmake_minimum_required(VERSION 2.8)
project( CaptureImage )
find_package( OpenCV REQUIRED )

add_executable( record src/video_write.cpp )
target_link_libraries( record ${OpenCV_LIBS} )

add_executable( analyze src/cv_analyze.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
