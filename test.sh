#!/bin/bash
#SBATCH --job-name=test_kc
#SBATCH --time=00:30:00
#SBATCH --output=test_output_%j.log

setup() {
    init
    kokkos
    printf "Setup complete!\n"
}

init() {
    printf "Setting up directories...\n"
    if [ -d $HOME/repos ]; then
        :
    else
        mkdir $HOME/repos
    fi

    if [ -d $HOME/install ]; then
        :
    else
        mkdir $HOME/install $HOME/install/kokkos
    fi
    printf "Complete!\n"
}

kokkos() {
    if [ -d $HOME/repos/kokkos ]; then
        printf "'kokkos' directory already exists."
    else
        git clone git@github.com:kokkos/kokkos.git $HOME/repos/kokkos -j16
    fi

    if [ -d $HOME/repos/kokkos/build ]; then
        rm -rf $HOME/repos/kokkos/build; mkdir $HOME/repos/kokkos/build
    else
        mkdir $HOME/repos/kokkos/build
    fi

    cd $HOME/repos/kokkos
    cmake \
        -S $HOME/repos/kokkos \
        -B $HOME/repos/kokkos/build \
        -DCMAKE_INSTALL_PREFIX=$HOME/install/kokkos \
        -DCMAKE_BUILD_TYPE=Release;
    cd build; make -j16; make install -j16
    printf "Kokkos setup complete!\n\n"
}

kokkos-comm(){
    if [ -d $HOME/repos/kokkos-comm/build ]; then
        rm -rf $HOME/repos/kokkos-comm/build
        mkdir $HOME/repos/kokkos-comm/build
    else
        mkdir $HOME/repos/kokkos-comm/build
    fi
    cmake -S $HOME/repos/kokkos-comm -B $HOME/repos/kokkos-comm/build -DCMAKE_CXX_COMPILER=g++ -DKokkos_ROOT=$HOME/install/kokkos
    make -C $HOME/repos/kokkos-comm/build #-j16
    printf "KokkosComm setup complete!\n\n"
}

clean() {
    printf "Cleaning up...\n"
    rm -rf $HOME/install/kokkos $HOME/repos/kokkos-comm/build
    printf "Cleanup complete!\n"
}

test(){
    cd $HOME/repos/kokkos-comm/build
    ctest -V
}

# Check for arguments
if [ "$1" == "clean" ]; then
    clean
elif [ "$1" == "test" ]; then
    test
elif [ "$1" == "setup" ]; then
    setup
elif [ "$1" == "all" ]; then 
    clean
    setup
    kokkos-comm
    test
else
    kokkos-comm
    test
fi