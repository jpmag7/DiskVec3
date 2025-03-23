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
#include <cstring>
#include <stack>


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
    void* mmap(void* addr, size_t length, uint64_t prot, uint64_t flags, uint64_t fd, off_t offset) {
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
    
    uint64_t munmap(void* addr, size_t length) {
        if (addr == NULL) {
            return -1;
        }
        if (!UnmapViewOfFile(addr)) {
            return -1;
        }
        return 0;
    }
    
    uint64_t msync(void* addr, size_t length, uint64_t flags) {
        if (FlushViewOfFile(addr, length)) {
            return 0;
        }
        return -1;
    }
    
    // Function to get file size in Windows
    off_t fsize(uint64_t fd) {
        struct _stat64 st;
        if (_fstat64(fd, &st) < 0) {
            return -1;
        }
        return st.st_size;
    }
    
    // Truncate function for Windows
    uint64_t ftruncate(uint64_t fd, off_t length) {
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



template <typename T>
std::tuple<T*, size_t, uint64_t> map_file(const std::string& filename, size_t file_size=0) {
    uint64_t fd;
    #ifdef _WIN32
    uint64_t flags = _O_RDWR | _O_BINARY;
    if (file_size > 0) {
        flags |= _O_CREAT | _O_TRUNC;
    }
    fd = ::_open(filename.c_str(), flags, _S_IREAD | _S_IWRITE);
    #else
    uint64_t flags = O_RDWR;
    if (file_size > 0) {
        flags |= O_CREAT | O_TRUNC;
    }
    fd = ::open(filename.c_str(), flags, 0666);
    #endif
    if (fd < 0) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    if (file_size > 0) {
        if (ftruncate(fd, file_size) < 0) {
            perror("Error truncating file");
            exit(EXIT_FAILURE);
        }
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("Error getting file size");
        exit(EXIT_FAILURE);
    }
    T* mapped = static_cast<T*>(mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    return {mapped, st.st_size, fd};
}


template <typename T>
void sync_file(T* mapped, uint64_t file_size) {
    if (msync(mapped, file_size, MS_SYNC) == -1) {
        perror("Error syncing memory to file");
        exit(EXIT_FAILURE);
    }
}


template <typename T>
void close_file(uint64_t fd, T* mapped, uint64_t file_size) {
    munmap(mapped, file_size);
    close(fd);
}


uint16_t float_to_half(float f);
float half_to_float(uint16_t h);


struct Half {
    uint16_t bits;
    Half() : bits(0) {}
    Half(float f) { bits = float_to_half(f); }
    operator float() const { return half_to_float(bits); }
};


uint16_t float_to_half(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = (x >> 16) & 0x8000;
    uint64_t exp = ((x >> 23) & 0xff) - 127 + 15;
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


struct NodeInfo {
    float threshold;
    uint64_t leftSize;    // number of nodes in left subtree
    uint64_t totalSize;   // total nodes in this subtree (including current node)
};




template <typename T>
void swapEmbAndVal(T* emb, uint32_t* vals, uint64_t idx1, uint64_t idx2, uint64_t dim) {
    for (uint64_t i = 0; i < dim; i++) {
        std::swap(emb[idx1 * dim + i], emb[idx2 * dim + i]);
    }
    std::swap(vals[idx1], vals[idx2]);
}


template <typename T>
float manhattanDistance(T* emb, uint64_t idx1, uint64_t idx2, uint64_t dim) {
    float dist = 0;
    uint64_t offset1 = idx1 * dim;
    uint64_t offset2 = idx2 * dim;
    // #pragma omp parallel for reduction(+:dist)
    for (uint64_t i = 0; i < dim; i++) {
        float a = static_cast<float>(emb[offset1 + i]);
        float b = static_cast<float>(emb[offset2 + i]);
        dist += std::fabs(a - b);
    }
    return dist;
}


template <typename T>
float manhattanDistance(T* emb, uint64_t idx, const std::vector<float>& query, uint64_t dim) {
    float dist = 0;
    uint64_t offset = idx * dim;
    // #pragma omp parallel for reduction(+:dist)
    for (uint64_t i = 0; i < dim; i++) {
        float a = static_cast<float>(emb[offset + i]);
        dist += std::fabs(a - query[i]);
    }
    return dist;
}



template <typename T>
void buildTreeInPlace(T* emb, uint32_t* vals, NodeInfo* nodeInfos, float* buffer, uint64_t dim, uint64_t count) {

    struct Frame {
        uint64_t start;
        uint64_t count;
    };

    std::stack<Frame> stack;
    stack.push({0, count});

    while (!stack.empty()) {

        Frame current = stack.top();
        stack.pop();
        uint64_t currStart = current.start;
        uint64_t currCount = current.count;
        
        if (currCount <= 0)
            continue;
        if (currCount == 1) {
            nodeInfos[currStart].threshold = 0;
            nodeInfos[currStart].leftSize = 0;
            nodeInfos[currStart].totalSize = 1;
            continue;
        }
        
        // Vantage point
        uint64_t pivot = currStart + (std::rand() % currCount);
        swapEmbAndVal(emb, vals, pivot, currStart, dim);
        uint64_t vpIndex = currStart;

        // Compute distances from each embedding (except vp) to the vantage point.
        for (uint64_t i = currStart + 1; i < currStart + currCount; i++) {
            buffer[i - currStart - 1] = manhattanDistance(emb, i, vpIndex, dim);
        }

        // Find median of these distances.
        uint64_t m = (currCount - 1) / 2;
        std::nth_element(buffer, buffer + m, buffer + currCount);
        float median = buffer[m];
        
        // Partition embeddings in [currStart, currStart+currCount) by median.
        uint64_t i = currStart + 1, j = currStart + currCount - 1;
        while (i <= j) {
            float d = manhattanDistance(emb, i, vpIndex, dim);
            if (d <= median) {
                i++;
            } else {
                swapEmbAndVal(emb, vals, i, j, dim);
                j--;
            }
        }

        // Create NodeInfo
        uint64_t leftCount = i - currStart - 1;
        nodeInfos[currStart].threshold = median;
        nodeInfos[currStart].leftSize = leftCount;
        nodeInfos[currStart].totalSize = currCount;
        
        stack.push({currStart + 1 + leftCount, currCount - 1 - leftCount});
        stack.push({currStart + 1, leftCount});
    }
}


template <typename T>
class VPTree {
public:
    VPTree() :  emb_fd(0), val_fd(0), tree_fd(0),
                emb_size(0), val_size(0), tree_size(0),
                emb(nullptr), vals(nullptr), nodeInfos(nullptr),
                num_points(0), dimension(0) {
        std::srand(std::time(nullptr));
    }
    ~VPTree() { close(); }

    bool build(const std::string& embedFile, const std::string& valueFile, const std::string& treeFile, uint64_t dim) {
        dimension = dim;
        
        std::tie(emb, emb_size, emb_fd) = map_file<T>(embedFile);
        num_points = emb_size / (dimension * sizeof(T));
        std::tie(vals, val_size, val_fd) = map_file<uint32_t>(valueFile);
        size_t treeSize = num_points * sizeof(NodeInfo);
        std::tie(nodeInfos, tree_size, tree_fd) = map_file<NodeInfo>(treeFile, treeSize);

        float* buffer = new float[num_points];
        buildTreeInPlace(emb, vals, nodeInfos, buffer, dimension, num_points);
        delete[] buffer;
        
        sync_file<T>(emb, emb_size);
        sync_file<uint32_t>(vals, val_size);
        sync_file<NodeInfo>(nodeInfos, tree_size);
        return true;
    }
    
    bool open(const std::string& embedFile, const std::string& valueFile, const std::string& treeFile, uint64_t dim) {
        dimension = dim;
        std::tie(emb, emb_size, emb_fd) = map_file<T>(embedFile);
        num_points = emb_size / (dimension * sizeof(T));
        std::tie(vals, val_size, val_fd) = map_file<uint32_t>(valueFile);
        size_t treeSize = num_points * sizeof(NodeInfo);
        std::tie(nodeInfos, tree_size, tree_fd) = map_file<NodeInfo>(treeFile);
        return true;
    }

    void close() {
        if (emb != nullptr) {
            close_file<T>(emb_fd, emb, emb_size);
            emb = nullptr;
        }
        if (vals != nullptr) {
            close_file<uint32_t>(val_fd, vals, val_size);
            vals = nullptr;
        }
        if (nodeInfos != nullptr) {
            close_file<NodeInfo>(tree_fd, nodeInfos, tree_size);
            nodeInfos = nullptr;
        }
    }

    uint64_t search(const std::vector<float>& query, float tau) {
        uint64_t bestIndex = -1;
        float bestDistance = std::numeric_limits<float>::infinity();
        searchRecursive(0, nodeInfos[0].totalSize, query, bestIndex, bestDistance, 1, tau);
        return vals[bestIndex];
    }

private:
    uint64_t emb_fd, val_fd, tree_fd;
    uint64_t emb_size, val_size, tree_size;
    T* emb;
    uint32_t* vals;
    NodeInfo* nodeInfos;
    size_t num_points;
    uint64_t dimension;

    void searchRecursive(uint64_t start, uint64_t count, const std::vector<float>& query, uint64_t &bestIndex, float &bestDistance, uint64_t depth, float tau) {
        if (count <= 0){
            return;
        }
        float d = manhattanDistance(emb, start, query, dimension);
        if (d < bestDistance) {
            bestDistance = d;
            bestIndex = start;
        }
        float diff = d - nodeInfos[start].threshold;
        uint64_t leftCount = nodeInfos[start].leftSize;
        uint64_t total = nodeInfos[start].totalSize;
        
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
        uint64_t dim = std::stoi(argv[5]);
        
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
        uint64_t dimVal = std::stoi(argv[5]); // argv[5] holds the dimension.
        uint64_t expectedArgs = 8 + dimVal; // 1+1+3+1+dim+1+1
        if (argc != expectedArgs) {
            std::cerr << "Search mode requires " << (7 + dimVal) << " arguments. diskvec search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]\n";
            return 1;
        }
        std::string embFile = argv[2];
        std::string valFile = argv[3];
        std::string treeFile = argv[4];
        uint64_t dim = dimVal;
        std::vector<float> query(dim);
        for (uint64_t i = 0; i < dim; i++) {
            query[i] = std::stof(argv[6 + i]); // Query elements from argv[6] to argv[6+dim-1]
        }
        float tau = std::stof(argv[6 + dim]); // Tau comes after the query elements.
        
        if (precision == "float32") {
            VPTree<float> tree;
            if (!tree.open(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to open VP Tree (float32).\n";
                return 1;
            }
            uint64_t best = tree.search(query, tau);
            if (best == -1)
                std::cout << "No neighbor found within tau.\n";
            else
                std::cout << "Nearest neighbor (float32) with value: " << best << "\n";
            tree.close();
        } else if (precision == "float16") {
            VPTree<Half> tree;
            if (!tree.open(embFile, valFile, treeFile, dim)) {
                std::cerr << "Failed to open VP Tree (float16).\n";
                return 1;
            }
            uint64_t best = tree.search(query, tau);
            if (best == -1)
                std::cout << "No neighbor found within tau.\n";
            else
                std::cout << "Nearest neighbor (float16) with value: " << best << "\n";
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