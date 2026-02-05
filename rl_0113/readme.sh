# step1 删除现有cpp路径下build目录

# step 2 重新编译cpp
# arm 64架构:
cd cpp
mkdir build
cd build
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 ..
make

# multi后缀是当前baseline，abo后缀是目前正在尝试的一版实验（直接输出补货量的）
