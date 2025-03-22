// Compile:
//  Bindings: python setup.py build_ext --inplace
//  Terminal: g++ -std=c++11 -O3 -o diskvec diskvec.cpp

// Enable POSIX features.
#define _POSIX_C_SOURCE 200809L

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <fstream>
#include <limits>
#include <cstring>     // For memcpy
#include <stack>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    #include <fcntl.h>
    
    // Define constants for Windows
    #define MAP_FAILED ((void*)-1)
    #define PROT_READ 1
    #define PROT_WRITE 2
    #define MAP_SHARED 1
    #define MS_SYNC 0
    
    // Windows implementation of mmap
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
        HANDLE hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) {
            return MAP_FAILED;
        }
    
        DWORD flProtect = 0;
        if ((prot & PROT_READ) && (prot & PROT_WRITE)) {
            flProtect = PAGE_READWRITE;
        } else if (prot & PROT_READ) {
            flProtect = PAGE_READONLY;
        } else {
            return MAP_FAILED;
        }
    
        DWORD dwDesiredAccess = 0;
        if ((prot & PROT_READ) && (prot & PROT_WRITE)) {
            dwDesiredAccess = FILE_MAP_WRITE;
        } else if (prot & PROT_READ) {
            dwDesiredAccess = FILE_MAP_READ;
        } else {
            return MAP_FAILED;
        }
    
        HANDLE hMapping = CreateFileMapping(hFile, NULL, flProtect, 0, 0, NULL);
        if (hMapping == NULL) {
            return MAP_FAILED;
        }
    
        void* mapped = MapViewOfFile(hMapping, dwDesiredAccess, 0, offset, length);
        CloseHandle(hMapping);
    
        if (mapped == NULL) {
            return MAP_FAILED;
        }
    
        return mapped;
    }
    
    int munmap(void* addr, size_t length) {
        if (addr == NULL) {
            return -1;
        }
        if (!UnmapViewOfFile(addr)) {
            return -1;
        }
        return 0;
    }
    
    int msync(void* addr, size_t length, int flags) {
        if (FlushViewOfFile(addr, length)) {
            return 0;
        }
        return -1;
    }
    
    // Function to get file size in Windows
    off_t fsize(int fd) {
        struct _stat64 st;
        if (_fstat64(fd, &st) < 0) {
            return -1;
        }
        return st.st_size;
    }
    
    // Truncate function for Windows
    int ftruncate(int fd, off_t length) {
        HANDLE hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile == INVALID_HANDLE_VALUE) {
            return -1;
        }
        
        LARGE_INTEGER li;
        li.QuadPart = length;
        
        if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) {
            return -1;
        }
        
        if (!SetEndOfFile(hFile)) {
            return -1;
        }
        
        return 0;
    }

    // For stat compatibility
    #define stat _stat64
    #define fstat _fstat64
    
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif






const size_t MMAP_CHUNK_SIZE = 1L * 1024 * 1024 * 1024; // 1GB chunks
template <typename T>
class MappedFile {
public:
    int fd;
    size_t file_size;
    std::vector<void*> mapped_chunks;
    size_t chunk_size;
    size_t num_chunks;
    
    MappedFile(const std::string& filename, size_t file_size = 0, size_t chunk_size = MMAP_CHUNK_SIZE) 
        : fd(-1), file_size(0), chunk_size(chunk_size) {  // Use member variable initialization
        
        #ifdef _WIN32
        int flags = _O_RDWR | _O_BINARY;
        if (file_size > 0) {
            flags |= _O_CREAT | _O_TRUNC;
        }
        fd = ::_open(filename.c_str(), flags, _S_IREAD | _S_IWRITE);
        #else
        int flags = O_RDWR;
        if (file_size > 0) {
            flags |= O_CREAT | O_TRUNC;
        }
        fd = ::open(filename.c_str(), flags, 0666);
        #endif

        if (fd < 0) {
            perror("Error opening file (mp1)");
            exit(EXIT_FAILURE);
        }

        // If file_size > 0, truncate it to the requested size
        if (file_size > 0) {
            if (ftruncate(fd, file_size) < 0) {
                perror("Error truncating file (mp2)");
                exit(EXIT_FAILURE);
            }
        }

        struct stat st;
        if (fstat(fd, &st) < 0) {
            perror("Error getting file size (mp3)");
            exit(EXIT_FAILURE);
        }

        this->file_size = st.st_size;
        num_chunks = (this->file_size + chunk_size - 1) / chunk_size;
        std::cout << "File size: " << this->file_size << " chunk_size: " << chunk_size << " num_chunks: " << num_chunks << std::endl;
        std::cout.flush();
        mapChunks();
    }

    void mapChunks() {
        for (size_t i = 0; i < num_chunks; i++) {
            size_t offset = i * chunk_size;
            size_t this_chunk_size = std::min<size_t>(chunk_size, file_size - offset);
            void* mapped = mmap(nullptr, this_chunk_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
            std::cout << "Mapped offset: " << offset << " chunk_size: " << chunk_size << " num_chunks: " << num_chunks << " this_chunk_size: " << this_chunk_size << std::endl;
            std::cout.flush();
            if (mapped == MAP_FAILED) {
                std::cout << this_chunk_size << " " << offset << " " << num_chunks << " " << i << std::endl;
                std::cout.flush();
                perror("Error mmap (mp4)");
                exit(EXIT_FAILURE);
            }
            mapped_chunks.push_back(mapped);
        }
    }

    void sync() {
        for (size_t i = 0; i < num_chunks; i++) {
            if (msync(mapped_chunks[i], std::min<size_t>(chunk_size, file_size - (i * chunk_size)), MS_SYNC) == -1) {
                perror("Error syncing memory to file (mp5)");
                exit(EXIT_FAILURE);
            }
        }
    }

    T& operator[](size_t idx) {
        size_t index = idx * sizeof(T);
        size_t chunk_index = index / chunk_size;
        size_t offset = (index % chunk_size) / sizeof(T);
        // std::cout << "mmap chk_idx: " << chunk_index << " offset: " << offset << " index: " << index << std::endl;
        // std::cout.flush();
        if (chunk_index >= num_chunks) {
            std::cout << "idx: " << idx << " chunk_size: " << chunk_index << " num_chunks: " << num_chunks << " offset: " << offset << " index: " << index << " type: " << typeid(T).name() << std::endl;
            std::cout.flush();
            throw std::out_of_range("Index out of mapped range (mp6)");
        }
        return static_cast<T*>(mapped_chunks[chunk_index])[offset];
    }

    ~MappedFile() {
        for (size_t i = 0; i < mapped_chunks.size(); i++) {
            munmap(mapped_chunks[i], std::min<size_t>(chunk_size, file_size - (i * chunk_size)));
        }
        close(fd);
    }
};




// Prototypes for half-precision conversion functions.
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

// ---------------------------
// Minimal half-precision type
// ---------------------------
struct Half {
    uint16_t bits;
    Half() : bits(0) {}
    Half(float f) { bits = float_to_half(f); }
    operator float() const { return half_to_float(bits); }
};

uint16_t float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xff) - 127 + 15;
    if(exp <= 0)
        return sign;
    else if(exp >= 31)
        return sign | 0x7c00;
    else {
        uint16_t mantissa = (x >> 13) & 0x3ff;
        return sign | (exp << 10) | mantissa;
    }
}

float half_to_float(uint16_t h) {
    uint16_t sign = h & 0x8000;
    uint16_t exp = (h >> 10) & 0x1f;
    uint16_t frac = h & 0x3ff;
    uint32_t f;
    if(exp == 0) {
        if(frac == 0) {
            f = sign << 16;
        } else {
            exp = 1;
            while((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3ff;
            f = (sign << 16) | ((exp - 1 + 127) << 23) | (frac << 13);
        }
    } else if(exp == 31) {
        f = (sign << 16) | 0x7f800000 | (frac << 13);
    } else {
        f = (sign << 16) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    return *reinterpret_cast<float*>(&f);
}

// ------------------------------
// NodeInfo: stored per node in tree file.
// We keep leftSize and totalSize to quickly compute subtree boundaries during search.
struct NodeInfo {
    float threshold;
    int leftSize;    // number of nodes in left subtree
    int totalSize;   // total nodes in this subtree (including current node)
};

template <typename T>
void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
void swapEmbAndVal(MappedFile<T>* emb, MappedFile<int32_t>* vals, int idx1, int idx2, int dim) {
    // Swap embeddings.
    // std::cout << "swap" << std::endl;
    // std::cout.flush();
    for (int i = 0; i < dim; i++) {
        swap<T>((*emb)[idx1 * dim + i], (*emb)[idx2 * dim + i]);
    }
    // Swap corresponding values.
    swap<int32_t>((*vals)[idx1], (*vals)[idx2]);
}

// ------------------------------
// Helper: compute Manhattan distance between two embeddings in array 'emb'.
// Optimized with OpenMP.
template <typename T>
float manhattanDistance(MappedFile<T>* emb, int idx1, int idx2, int dim) {
    // std::cout << "idx1: " << idx1 << " idx2: " << idx2 << std::endl;
    // std::cout.flush();
    float dist = 0;
    int offset1 = idx1 * dim;
    int offset2 = idx2 * dim;
    // std::cout << "offset1: " << offset1 << " offset2: " << offset2 << std::endl;
    // std::cout.flush();
    //#pragma omp parallel for reduction(+:dist)
    for (int i = 0; i < dim; i++) {
        float a = static_cast<float>((*emb)[offset1 + i]);
        float b = static_cast<float>((*emb)[offset2 + i]);
        dist += std::fabs(a - b);
    }
    return dist;
}

// ------------------------------
// Helper: compute Manhattan distance between an embedding and a query vector.
template <typename T>
float manhattanDistance(MappedFile<T>* emb, int idx, const std::vector<float>& query, int dim) {
    // std::cout << "idx: " << idx << std::endl;
    // std::cout.flush();
    float dist = 0;
    int offset = idx * dim;
    // std::cout << "offset: " << offset << std::endl;
    // std::cout.flush();
    //#pragma omp parallel for reduction(+:dist)
    for (int i = 0; i < dim; i++) {
        float a = static_cast<float>((*emb)[offset + i]);
        dist += std::fabs(a - query[i]);
    }
    return dist;
}

// ------------------------------
// Recursive function that builds the VP-tree in place by reordering the embeddings
// and the corresponding values (in the range [start, start+count)).
// It writes node info into 'nodeInfos'. 'buffer' is a temporary float array.
template <typename T>
int buildTreeInPlace1(MappedFile<T> emb, int32_t* vals, int start, int count, int dim, NodeInfo* nodeInfos, float* buffer) {
    if (count <= 0)
        return 0;
    if (count == 1) {
        nodeInfos[start].threshold = 0;
        nodeInfos[start].leftSize = 0;
        nodeInfos[start].totalSize = 1;
        return 1;
    }
    // Choose a random vantage point from current segment.
    int pivot = start + (std::rand() % count);
    // Swap chosen vantage point to the end (reordering both embeddings and values).
    swapEmbAndVal(emb, vals, pivot, start, dim);
    int vpIndex = start;
    // Compute distances from each embedding (except vp) to the vantage point.
    //#pragma omp parallel for
    for (int i = start + 1; i < start + count; i++) {
        buffer[i - start - 1] = manhattanDistance(emb, i, vpIndex, dim);
    }
    // Find median of these distances.
    int m = (count - 1) / 2;
    std::nth_element(buffer, buffer + m + 1, buffer + count);
    float median = buffer[m + 1];

    // Partition embeddings in [start, start+count-1) by median.
    int i = start + 1, j = start + count - 1, d = 0, swapBuff = 0;
    while (i <= j) {
        //float d = manhattanDistance(emb, i, vpIndex, dim);
        d = buffer[i - start - 1];
        if (d <= median) {
            i++;
        } else {
            swapBuff = buffer[j - start];
            buffer[j - start] = buffer[i - start - 1];
            buffer[i - start - 1] = swapBuff;
            swapEmbAndVal(emb, vals, i, j, dim);
            j--;
        }
    }
    int leftCount = i - start - 1;
    // Swap the vantage point (currently at vpIndex) with the element at 'start'.
    //swapEmbAndVal(emb, vals, start, vpIndex, dim);
    // Recurse: left subtree in [start+1, start+1+leftCount), right subtree in remaining segment.
    int leftSize = buildTreeInPlace(emb, vals, start + 1, leftCount, dim, nodeInfos, buffer);
    int rightSize = buildTreeInPlace(emb, vals, start + 1 + leftCount, count - 1 - leftCount, dim, nodeInfos, buffer);
    int totalSize = 1 + leftSize + rightSize;
    nodeInfos[start].threshold = median;
    nodeInfos[start].leftSize = leftSize;
    nodeInfos[start].totalSize = totalSize;
    return totalSize;
}







template <typename T>
void buildTreeInPlace2(MappedFile<T> emb, int32_t* vals, int start, int count, int dim, NodeInfo* nodeInfos, float* buffer) {
    if (count <= 0)
        return;
    if (count == 1) {
        nodeInfos[start].threshold = 0;
        nodeInfos[start].leftSize = 0;
        nodeInfos[start].totalSize = 1;
        return;
    }
    // Choose a random vantage point from current segment.
    int pivot = start + (std::rand() % count);
    // Swap chosen vantage point to the end (reordering both embeddings and values).
    swapEmbAndVal(emb, vals, pivot, start, dim);
    int vpIndex = start;
    // Compute distances from each embedding (except vp) to the vantage point.
    //#pragma omp parallel for
    for (int i = start + 1; i < start + count; i++) {
        buffer[i - start - 1] = manhattanDistance(emb, i, vpIndex, dim);
    }
    // Find median of these distances.
    int m = (count - 1) / 2;
    std::nth_element(buffer, buffer + m + 1, buffer + count);
    float median = buffer[m + 1];

    // Partition embeddings in [start, start+count-1) by median.
    int i = start + 1, j = start + count - 1, d = 0;
    while (i <= j) {
        d = buffer[i - start - 1];
        if (d <= median) {
            i++;
        } else {
            std::swap(buffer[j - start], buffer[i - start - 1]);
            swapEmbAndVal(emb, vals, i, j, dim);
            j--;
        }
    }
    int leftCount = i - start - 1;
    nodeInfos[start].threshold = median;
    nodeInfos[start].leftSize = leftCount;
    nodeInfos[start].totalSize = count;
    buildTreeInPlace(emb, vals, start + 1, leftCount, dim, nodeInfos, buffer);
    buildTreeInPlace(emb, vals, start + 1 + leftCount, count - 1 - leftCount, dim, nodeInfos, buffer);
}






template <typename T>
void buildTreeInPlace(MappedFile<T>* emb, MappedFile<int32_t>* vals, MappedFile<NodeInfo>* nodeInfos, float* buffer, int dim, int count) {

    struct Frame {
        int start;
        int count;
    };

    std::stack<Frame> stack;
    stack.push({0, count});

    // std::cout << "tree..." << std::endl;
    // std::cout.flush();
    
    while (!stack.empty()) {

        // std::cout << "tree..." << std::endl;
        // std::cout.flush();
        
        Frame current = stack.top();
        stack.pop();
        int currStart = current.start;
        int currCount = current.count;
        
        if (currCount <= 0)
            continue;
        if (currCount == 1) {
            (*nodeInfos)[currStart].threshold = 0;
            (*nodeInfos)[currStart].leftSize = 0;
            (*nodeInfos)[currStart].totalSize = 1;
            continue;
        }
        
        // Choose a random vantage point from current segment.
        int pivot = currStart + (std::rand() % currCount);
        // Swap chosen vantage point to the beginning (or any designated location)
        swapEmbAndVal(emb, vals, pivot, currStart, dim);
        int vpIndex = currStart;

        // std::cout << "0" << std::endl;
        // std::cout.flush();
        
        // Compute distances from each embedding (except vp) to the vantage point.
        for (int i = currStart + 1; i < currStart + currCount; i++) {
            // std::cout << "i: " << i << std::endl;
            // std::cout.flush();
            buffer[i - currStart - 1] = manhattanDistance(emb, i, vpIndex, dim);
        }

        // std::cout << "1" << std::endl;
        // std::cout.flush();
        
        // Find median of these distances.
        int m = (currCount - 1) / 2;
        std::nth_element(buffer, buffer + m, buffer + currCount);
        float median = buffer[m];
        
        // Partition embeddings in [currStart, currStart+currCount) by median.
        int i = currStart + 1, j = currStart + currCount - 1;
        while (i <= j) {
            float d = manhattanDistance(emb, i, vpIndex, dim);//buffer[i - currStart - 1];
            if (d <= median) {
                i++;
            } else {
                // std::swap(buffer[j - currStart], buffer[i - currStart - 1]);
                swapEmbAndVal(emb, vals, i, j, dim);
                j--;
            }
        }
        int leftCount = i - currStart - 1;
        (*nodeInfos)[currStart].threshold = median;
        (*nodeInfos)[currStart].leftSize = leftCount;
        (*nodeInfos)[currStart].totalSize = currCount;
        
        stack.push({currStart + 1 + leftCount, currCount - 1 - leftCount});
        stack.push({currStart + 1, leftCount});
    }
}






// ------------------------------
// VP-Tree class that reorders embeddings and values in place.
// The tree (node info) is stored in a separate file ("tree.dat").
// Precision: use VPTree<float> for 32-bit data or VPTree<Half> for 16-bit data.
// (The caller must choose the proper instantiation.)
template <typename T>
class VPTree {
public:
    VPTree() : embed_fd(-1), val_fd(-1), thresh_fd(-1), emb(nullptr), vals(nullptr), nodeInfos(nullptr),
               num_points(0), dimension(0) {
        std::srand(std::time(nullptr));
    }
    ~VPTree() { close(); }


    bool build(const std::string& embedFile, const std::string& valueFile, const std::string& treeFile, int dim) {
        std::cout << "Building tree..." << std::endl;
        std::cout.flush();
        
        dimension = dim;
        
        emb = new MappedFile<T>(embedFile);
        num_points = emb->file_size / (dimension * sizeof(T));

        std::cout << "Building tree..." << std::endl;
        std::cout.flush();

        vals = new MappedFile<int32_t>(valueFile);

        std::cout << "Building tree..." << std::endl;
        std::cout.flush();

        size_t treeSize = num_points * sizeof(NodeInfo);
        nodeInfos = new MappedFile<NodeInfo>(treeFile, treeSize);

        std::cout << "Building tree..." << std::endl;
        std::cout.flush();

        float* buffer = new float[num_points];
        buildTreeInPlace(emb, vals, nodeInfos, buffer, dimension, num_points);
        
        std::cout << "Building tree..." << std::endl;
        std::cout.flush();

        delete[] buffer;
        emb->sync();
        vals->sync();
        nodeInfos->sync();

        return true;
    }

    
    bool open(const std::string& embedFile, const std::string& valueFile, const std::string& treeFile, int dim) {
        dimension = dim;
        emb = new MappedFile<T>(embedFile);
        num_points = emb->file_size / (dimension * sizeof(T));
        vals = new MappedFile<int32_t>(valueFile);
        size_t treeSize = num_points * sizeof(NodeInfo);
        nodeInfos = new MappedFile<NodeInfo>(treeFile);
        return true;
    }

    void close() {
        if (emb != nullptr) {
            delete emb;
            emb = nullptr;
        }
        if (vals != nullptr) {
            delete vals;
            vals = nullptr;
        }
        if (nodeInfos != nullptr) {
            delete nodeInfos;
            nodeInfos = nullptr;
        }
    }

    int search(const std::vector<float>& query, float tau) {
        int bestIndex = -1;
        float bestDistance = std::numeric_limits<float>::infinity();
        searchRecursive(0, (*nodeInfos)[0].totalSize, query, bestIndex, bestDistance, 1, tau);
        return (*vals)[bestIndex];
    }

private:
    int embed_fd, val_fd, thresh_fd;
    MappedFile<T>* emb;
    MappedFile<int32_t>* vals;
    MappedFile<NodeInfo>* nodeInfos;
    size_t num_points;
    int dimension;

    void searchRecursive(int start, int count, const std::vector<float>& query, int &bestIndex, float &bestDistance, int depth, float tau) {
        if (count <= 0){
            return;
        }
        float d = manhattanDistance(emb, start, query, dimension);
        if (d < bestDistance) {
            bestDistance = d;
            bestIndex = start;
        }
        float diff = d - (*nodeInfos)[start].threshold;
        int leftCount = (*nodeInfos)[start].leftSize;
        int total = (*nodeInfos)[start].totalSize;
        
        if (diff < 0) {
            searchRecursive(start + 1, leftCount, query, bestIndex, bestDistance, depth+1, tau);
            float prob = std::max<float>(1.0f, std::log2((float)depth)) * tau;
            if (std::fabs(diff) < bestDistance * prob)
                searchRecursive(start + 1 + leftCount, total - 1 - leftCount, query, bestIndex, bestDistance, depth+1, tau);
        } else {
            searchRecursive(start + 1 + leftCount, total - 1 - leftCount, query, bestIndex, bestDistance, depth+1, tau);
            float prob = std::max<float>(1.0f, std::log2((float)depth)) * tau;
            if (std::fabs(diff) < bestDistance * prob)
                searchRecursive(start + 1, leftCount, query, bestIndex, bestDistance, depth+1, tau);
        }
    }
};


#ifndef BUILD_PYBINDINGS

int main(int argc, char** argv) {
    // Usage:
    // Build:   build [emb_file] [val_file] [dim] [precision]
    // Search:  search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]
    if (argc < 6) {
        std::cout << "Usage:\n"
                  << "  Build:  " << argv[0] << " build [emb_file] [val_file] [dim] [precision]\n"
                  << "  Search: " << argv[0] << " search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]\n";
        return 1;
    }
    
    std::string mode = argv[1];
    // The last argument is used to specify the precision.
    std::string precision = argv[argc - 1];
    
    if (mode == "build") {
        if (argc != 7) {
            std::cerr << "Build mode requires 4 arguments: [emb_file] [val_file] [tree_file] [dim] [precision]\n";
            return 1;
        }
        std::string embFile = argv[2];
        std::string valFile = argv[3];
        std::string treeFile = argv[4];
        int dim = std::stoi(argv[5]);
        
        if (precision == "float32") {
            VPTree<float> tree;
            if (!tree.build(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to build VP Tree in place (float32).\n";
                return 1;
            }
            tree.close();
        } else if (precision == "float16") {
            VPTree<Half> tree;
            if (!tree.build(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to build VP Tree in place (float16).\n";
                return 1;
            }
            tree.close();
        } else {
            std::cerr << "Unknown precision: " << precision << "\n";
            return 1;
        }
        std::cout << "VP Tree built successfully.\n";
        
    } else if (mode == "search") {
        // Expected arguments:
        // search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]
        // Total arguments = 1(mode) + 3(files) + 1(dim) + (dim query elements) + 1(tau) + 1(precision)
        int dimVal = std::stoi(argv[5]); // argv[5] holds the dimension.
        int expectedArgs = 8 + dimVal; // 1+1+3+1+dim+1+1
        if (argc != expectedArgs) {
            std::cerr << "Search mode requires " << (7 + dimVal) << " arguments. diskvec search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]\n";
            return 1;
        }
        std::string embFile = argv[2];
        std::string valFile = argv[3];
        std::string treeFile = argv[4];
        int dim = dimVal;
        std::vector<float> query(dim);
        for (int i = 0; i < dim; i++) {
            query[i] = std::stof(argv[6 + i]); // Query elements from argv[6] to argv[6+dim-1]
        }
        float tau = std::stof(argv[6 + dim]); // Tau comes after the query elements.
        
        if (precision == "float32") {
            VPTree<float> tree;
            if (!tree.open(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to open VP Tree (float32).\n";
                return 1;
            }
            int best = tree.search(query, tau);
            if (best == -1)
                std::cout << "No neighbor found within tau.\n";
            else
                std::cout << "Nearest neighbor (float32) at index: " << best
                          << " with associated value: (retrieve from values file)" << "\n";
            tree.close();
        } else if (precision == "float16") {
            VPTree<Half> tree;
            if (!tree.open(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to open VP Tree (float16).\n";
                return 1;
            }
            int best = tree.search(query, tau);
            if (best == -1)
                std::cout << "No neighbor found within tau.\n";
            else
                std::cout << "Nearest neighbor (float16) at index: " << best
                          << " with associated value: (retrieve from values file)" << "\n";
            tree.close();
        } else {
            std::cerr << "Unknown precision: " << precision << "\n";
            return 1;
        }
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
    }
    return 0;
}

#endif // BUILD_PYBINDINGS

#ifdef BUILD_PYBINDINGS
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// In Python, you choose the proper class based on your data precision.
PYBIND11_MODULE(diskvec, m) {
    m.doc() = "VP Tree module with in-place reordering of embeddings and values. "
              "Use VPTreeFloat for float32 data and VPTreeHalf for float16 data.";
    
    py::class_<VPTree<float>>(m, "VPTreeFloat")
        .def(py::init<>())
        .def("build", &VPTree<float>::build,
             "Build VP tree in place from embeddings and values files (float32)",
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
        .def("open", &VPTree<float>::open,
             "Open existing VP tree (float32)",
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
        .def("close", &VPTree<float>::close)
        .def("search", &VPTree<float>::search,
             "Search VP tree (float32)", py::arg("query"), py::arg("tau"));
    
    py::class_<VPTree<Half>>(m, "VPTreeHalf")
        .def(py::init<>())
        .def("build", &VPTree<Half>::build,
             "Build VP tree in place from embeddings and values files (float16)",
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
        .def("open", &VPTree<Half>::open,
             "Open existing VP tree (float16)",
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
        .def("close", &VPTree<Half>::close)
        .def("search", &VPTree<Half>::search,
             "Search VP tree (float16)", py::arg("query"), py::arg("tau"));
}
#endif
