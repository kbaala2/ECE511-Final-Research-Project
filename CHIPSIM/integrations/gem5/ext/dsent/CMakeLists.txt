# Copyright (c) 2014 Mark D. Hill and David A. Wood
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 2.6)

project (DSENT)
# find out all the file that need to be compiled
file(GLOB_RECURSE SRCS  "*.h" "*.cc")

# add the library
add_library(dsent MODULE ${SRCS})

SET(PYTHON_CONFIG "python3-config")

execute_process(COMMAND ${PYTHON_CONFIG} --includes OUTPUT_VARIABLE PYTHON_INCLUDE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_CONFIG} --cflags OUTPUT_VARIABLE _PYTHON_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_CONFIG} --ldflags OUTPUT_VARIABLE PYTHON_LFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

# Remove -I flags from _PYTHON_CFLAGS, as include paths will be handled by PYTHON_INCLUDE_FLAGS
string(REGEX REPLACE "-I[^ ]+ *" "" PYTHON_CFLAGS "${_PYTHON_CFLAGS}")

# Add CMAKE_SOURCE_DIR to include_directories for local headers
include_directories("${CMAKE_SOURCE_DIR}")

set_target_properties(dsent PROPERTIES
        PREFIX ""
        # Add Python include flags and cleaned Python CFLAGS to the compiler flags
        COMPILE_FLAGS "${PYTHON_INCLUDE_FLAGS} ${PYTHON_CFLAGS} -std=c++11"
        LINK_FLAGS "${PYTHON_LFLAGS}")
