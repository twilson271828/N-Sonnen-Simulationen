# Copyright (c) 2019-2024, Sascha Willems
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.22)
include(cmake/sample_helper.cmake)

get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
get_filename_component(CATEGORY_NAME ${PARENT_DIR} NAME)

add_subdirectory(src)
add_subdirectory(include)


add_sample_with_tags(
    ID ${FOLDER_NAME}
    CATEGORY ${CATEGORY_NAME}
    AUTHOR "Sascha Willems"
    NAME "Compute N-Body simulation"
    DESCRIPTION "Multi-pass compute dispatch N-Body particle simulation"
    SHADER_FILES_GLSL
        "glsl/particle.vert"
        "glsl/particle.frag"
        "glsl/particle_calculate.comp"
        "glsl/particle_integrate.comp"
    SHADER_FILES_HLSL
        "hlsl/particle.vert.hlsl"
        "hlsl/particle.frag.hlsl"
        "hlsl/particle_calculate.comp.hlsl"
        "hlsl/particle_integrate.comp.hlsl")
