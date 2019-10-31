clang++ -DIODIRECTENABLE -O3 -Wall -Wextra -o remote_thr_direct_file remote_thr_file.cpp -L/home/bicer/local/lib -I/home/bicer/local/include  -lzmq
clang++ -DIOFILEENABLE -O3 -Wall -Wextra -o remote_thr_file remote_thr_file.cpp -L/home/bicer/local/lib -I/home/bicer/local/include  -lzmq
