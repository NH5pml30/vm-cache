cmake_minimum_required(VERSION 3.0.0)
project(cache VERSION 0.1.0 LANGUAGES C CXX)

add_executable(cache main.cpp)
set_property(TARGET cache PROPERTY CXX_STANDARD 20)
