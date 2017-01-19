# FastCCD

This is a fast coordinate descent algorithm for matrix factorization.

### Usage
* g++ -L -lpthread -fopenmp  -o fastccd *h *cpp
* ./fastccd  -n 4 ./toy-example/

### Time Complexity
* The time complexity is O(|nnz|k), where |nnz| is the number of non-zero values.

### Reference
* Some codes are from http://www.cs.utexas.edu/~rofuyu/libpmf/. Thanks.
