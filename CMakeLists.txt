cmake_minimum_required(VERSION 3.31)
project(RLTemplate)

set(CMAKE_CXX_STANDARD 20)

include_directories(/path/to/eigen)

add_executable(RLTemplate
        NeuralNetwork.cpp
        NeuralNetwork.h
#        clothesClassification.cpp
        radiusClassification.cpp
)
