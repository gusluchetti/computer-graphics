cmake_minimum_required(VERSION 3.22)
project(opengl_test)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES main.cpp ${PROJECT_SOURCE_DIR}/src/glad.c)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} SDL2 dl)
