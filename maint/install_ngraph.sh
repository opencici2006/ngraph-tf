#!/bin/bash

# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ******************************************************************************

# TODO: remove these
#declare _INSTALL_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
#declare _INSTALL_SCRIPT_DIR="$( cd $(dirname "${_INSTALL_SCRIPT_NAME}") && pwd )"
#source "${_INSTALL_SCRIPT_DIR}/bash_lib.sh"

exit_now() {
    echo $1 >&2
    exit 1
}

check_dir_exists(){
    local DIRNAME="${1}"
    if [ -d "$DIRNAME" ]; then
        echo "1"
    else
        echo "0"
    fi
}

prepare_venv() {
    # Check and read arguments
    if (( $# != 2 )); then
        exit_now "Usage: ${FUNCNAME[0]} <venv-path> <python-version-2-or-3>. Expected 2, got $# arguments"
    fi
    local VENV_LOC="${1}"
    local REQUESTED_PYTHON_VERSION="${2}"

    if [[ $(check_dir_exists "${VENV_LOC}") == "1" ]]; then
        return
    fi

    # Create virtual environment for Python 2 or 3
    if [[ "${REQUESTED_PYTHON_VERSION}" == "3" ]]; then
        virtualenv --system-site-packages -p python3 ${VENV_LOC}
    else
        if [[ "${REQUESTED_PYTHON_VERSION}" == "2" ]]; then
            virtualenv --system-site-packages -p /usr/bin/python2 ${VENV_LOC}
        else
            exit_now "Expected Python version 2 or 3 but got ${REQUESTED_PYTHON_VERSION}"
        fi
    fi

    # Install required libraries
    # TODO: make this optional, In option 1, when we pip install tensorflow, these might already install
    ${VENV_LOC}/bin/pip install numpy mock keras
}

set_tf_configure_options(){
    if (( $# != 1 )); then
        exit_now "Usage: ${FUNCNAME[0]} <option>. Expected 1, got $# arguments"
    fi
    local NGTF_BUILD_OPTION="${1}"

    export TF_NEED_AWS=0
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_JEMALLOC=0
    export TF_NEED_KAFKA=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_COMPUTECPP=0
    export TF_NEED_OPENCL=0
    export TF_CUDA_CLANG=0
    export TF_NEED_TENSORRT=0
    export TF_DOWNLOAD_CLANG=0
    export TF_ENABLE_XLA=0
    export TF_NEED_GDR=0
    export TF_NEED_VERBS=0
    export TF_NEED_MPI=0
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_NEED_CUDA=0

    if [[ $NGTF_BUILD_OPTION == "3" ]]; then
        export TF_NEED_NGRAPH=1
    else
        if [[ $NGTF_BUILD_OPTION == "2" ]]; then
            export TF_NEED_NGRAPH=0
        else
            exit_now "Expected option to be 2 or 3, but got ${NGTF_BUILD_OPTION}"
        fi
    fi

    # Use this flag in skylake machines only
    export CC_OPT_FLAGS="-march=broadwell"
}

setup_bazel(){
    if (( $# != 0 )); then
        exit_now "Usage: ${FUNCNAME[0]}. Expected 0, got $# arguments"
    fi
    declare BAZEL_PROG
    if ! BAZEL_PROG="$(command -v bazel)"; then
        echo "Bazel not found. Downloading and installing"
        wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh      
        chmod +x bazel-0.16.0-installer-linux-x86_64.sh
        ./bazel-0.16.0-installer-linux-x86_64.sh --user
        export PATH=$PATH:~/bin
        rm bazel-0.16.0-installer-linux-x86_64.sh
    fi
    # TODO: If bazel version | grep "Build label" returns incompatible bazel version, reinstall it.
}

download_tf(){
    # Check and read arguments
    declare TF_TAG
    # TODO: always selecting master. debug that
    if (( $# == 1 )); then
        local TF_TAG="${1}"
    else
        if (( $# == 0 )); then
            local TF_TAG="master"
        else
            exit_now "Usage: ${FUNCNAME[0]} <optional tf branch or tag (default: master)>. Expected 0 or 1, got $# arguments"
        fi
    fi

    if [[ $(check_dir_exists "tensorflow") == "1" ]]; then
        return
    fi

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    git checkout $(TF_TAG)
    git status
    cd ..
}

download_ngtf(){
    # Check and read arguments
    declare NGTF_TAG
    if (( $# == 1 )); then
        local NGTF_TAG="${1}"
    else
        if (( $# == 0 )); then
            local NGTF_TAG="master"
        else
            exit_now "Usage: ${FUNCNAME[0]} <optional tf branch or tag (default: master)>. Expected 0 or 1, got $# arguments"
        fi
    fi

    if [[ $(check_dir_exists "ngraph-tf") == "1" ]]; then
        return
    fi

    git clone https://github.com/NervanaSystems/ngraph-tf.git
    cd ngraph-tf
    git checkout ${NGTF_TAG}
    git status
    cd ..
}

# Main script starts here
NGTF_BUILD_OPTION=$1  #"2"  # Must be 1, 2 or 3
VENV_LOC=$2
FULL_VENV_PATH=$(realpath ${VENV_LOC})
PY_VERSION=$3
NGTF_TAG=$4
if [[ ${NGTF_BUILD_OPTION} != 1 ]]; then
    TF_DL_DIR=$5
    FULL_TF_PATH=$(realpath ${TF_DL_DIR})
    TF_TAG=$6
    NGTF_DIR_REL_TO_TF=${TF_DL_DIR}/..
    FULL_NGTF_PATH=$(realpath ${NGTF_DIR_REL_TO_TF})
fi
# sample usage:
# ./install_ngraph.sh 2 venvtest 2 master my_tf_dl v1.11.0-rc2


prepare_venv ${FULL_VENV_PATH} ${PY_VERSION}

# Prepare TF
if [[ ${NGTF_BUILD_OPTION} == 1 ]]; then
    ${FULL_VENV_PATH}/bin/pip install tensorflow==1.11.0rc2
else
    setup_bazel
    mkdir ${TF_DL_DIR}
    cd ${TF_DL_DIR}
    download_tf ${TF_TAG}

    # make and install TF
    set_tf_configure_options "${NGTF_BUILD_OPTION}"
    cd tensorflow
    export PYTHON_BIN_PATH=${FULL_VENV_PATH}/bin/python
    #TODO: in case of py 3, this should be py3.6 etc
    export PYTHON_LIB_PATH=${FULL_VENV_PATH}/lib/python2.7/site-packages
    ./configure
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ./
    ${FULL_VENV_PATH}/bin/pip install -U ./tensorflow-1.*whl
fi

if [[ ${NGTF_BUILD_OPTION} == 1 || ${NGTF_BUILD_OPTION} == 2 ]]; then
    # make and build ngraph-tf
    cd ${FULL_NGTF_PATH}
    download_ngtf ${NGTF_TAG}
    cd ngraph-tf
    mkdir build
    cd build
    if [[ ${NGTF_BUILD_OPTION} == 1 ]]; then
        cmake ..
    else
        if [[ ${NGTF_BUILD_OPTION} == 2 ]]; then
            cmake -DUNIT_TEST_ENABLE=TRUE -DTF_SRC_DIR=${FULL_TF_PATH} ..
        fi
    fi
    make -j
    make install
    ${FULL_VENV_PATH}/bin/pip install python/dist/ngraph-0.6.0-py2.py3-none-linux_x86_64.whl
fi

#if [[ ${NGTF_BUILD_OPTION} == 2 ]]; then
#   cd test
#   ./gtest
#fi

echo "To start the virtual environment: source ${VENV_LOC}/bin/activate"

