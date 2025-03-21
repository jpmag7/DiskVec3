// Compile:
//  Bindings: python setup.py build_ext --inplace

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

// ------------------------------
// Helper: swap one embedding and its corresponding value.
// 'emb' is a pointer to the embeddings array (of type T),
// 'vals' is a pointer to the values array (int32_t).
// Each embedding has 'dim' elements.
template <typename T>
void swapEmbAndVal(T* emb, int32_t* vals, int idx1, int idx2, int dim) {
    // Swap embeddings.
    for (int i = 0; i < dim; i++) {
        std::swap(emb[idx1 * dim + i], emb[idx2 * dim + i]);
    }
    // Swap corresponding values.
    std::swap(vals[idx1], vals[idx2]);
}

// ------------------------------
// Helper: compute Manhattan distance between two embeddings in array 'emb'.
// Optimized with OpenMP.
template <typename T>
float manhattanDistance(T* emb, int idx1, int idx2, int dim) {
    float dist = 0;
    int offset1 = idx1 * dim;
    int offset2 = idx2 * dim;
    //#pragma omp parallel for reduction(+:dist)
    for (int i = 0; i < dim; i++) {
        float a = static_cast<float>(emb[offset1 + i]);
        float b = static_cast<float>(emb[offset2 + i]);
        dist += std::fabs(a - b);
    }
    return dist;
}

// ------------------------------
// Helper: compute Manhattan distance between an embedding and a query vector.
template <typename T>
float manhattanDistance(T* emb, int idx, const std::vector<float>& query, int dim) {
    float dist = 0;
    int offset = idx * dim;
    //#pragma omp parallel for reduction(+:dist)
    for (int i = 0; i < dim; i++) {
        float a = static_cast<float>(emb[offset + i]);
        dist += std::fabs(a - query[i]);
    }
    return dist;
}

// ------------------------------
// Recursive function that builds the VP-tree in place by reordering the embeddings
// and the corresponding values (in the range [start, start+count)).
// It writes node info into 'nodeInfos'. 'buffer' is a temporary float array.
template <typename T>
int buildTreeInPlace(T* emb, int32_t* vals, int start, int count, int dim, NodeInfo* nodeInfos, float* buffer) {
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
    for (int i = start+1; i < start + count; i++) {
        buffer[i - start - 1] = manhattanDistance(emb, i, vpIndex, dim);
    }
    // Find median of these distances.
    int m = (count - 1) / 2;
    std::nth_element(buffer, buffer + m + 1, buffer + count);
    float median = buffer[m + 1];

    // Partition embeddings in [start, start+count-1) by median.
    int i = start + 1, j = start + count - 1;
    while (i <= j) {
        float d = manhattanDistance(emb, i, vpIndex, dim);
        if (d <= median) {
            i++;
        } else {
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

    // Build: Reorder the embeddings file and the corresponding values file in place,
    // and write node info into "tree.dat".
    // 'embedFile' and 'valueFile' are the filenames for embeddings and values.
    bool build(const std::string& embedFile, const std::string& valueFile, int dim) {
        std::cerr << "Building..." << std::endl;
        std::cerr.flush();
        dimension = dim;
        // Open embeddings file (read-write).
        #ifdef _WIN32
        embed_fd = ::_open(embedFile.c_str(), _O_RDWR | _O_BINARY);
        #else
        embed_fd = ::open(embedFile.c_str(), O_RDWR);
        #endif
        std::cerr << "Embedding file opened" << std::endl;
        std::cerr.flush();
        if (embed_fd < 0) {
            std::cerr << "Error opening embeddings file: " << embedFile << std::endl;
            std::cerr.flush();
            return false;
        }
        struct stat st;
        if (fstat(embed_fd, &st) < 0) {
            std::cerr << "Error fstat on embeddings file." << std::endl;
            std::cerr.flush();
            return false;
        }
        size_t fileSize = st.st_size;
        num_points = fileSize / (dimension * sizeof(T));
        std::cerr << "Mapping embedding memory " << num_points << std::endl;
        std::cerr.flush();
        emb = (T*) mmap(nullptr, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, embed_fd, 0);
        std::cerr << "Embedding memory mapped " << num_points << std::endl;
        std::cerr.flush();
        if (emb == MAP_FAILED) {
            std::cerr << "Error mmapping embeddings file." << std::endl;
            std::cerr.flush();
            return false;
        }
        // Open values file (read-write). Assumed to be int32_t per embedding.
        #ifdef _WIN32
        val_fd = ::_open(valueFile.c_str(), _O_RDWR | _O_BINARY);
        #else
        val_fd = ::open(valueFile.c_str(), O_RDWR);
        #endif
        std::cerr << "Opening values file" << std::endl;
        std::cerr.flush();
        if (val_fd < 0) {
            std::cerr << "Error opening values file: " << valueFile << std::endl;
            std::cerr.flush();
            return false;
        }
        struct stat vst;
        if (fstat(val_fd, &vst) < 0) {
            std::cerr << "Error fstat on values file." << std::endl;
            std::cerr.flush();
            return false;
        }
        size_t valSize = vst.st_size;
        if (valSize != num_points * sizeof(int32_t)) {
            std::cerr << "Mismatch between embeddings and values count." << std::endl;
            std::cerr.flush();
            return false;
        }
        vals = (int32_t*) mmap(nullptr, valSize, PROT_READ | PROT_WRITE, MAP_SHARED, val_fd, 0);
        if (vals == MAP_FAILED) {
            std::cerr << "Error mmapping values file." << std::endl;
            std::cerr.flush();
            return false;
        }
        // Create thresholds file ("tree.dat").
        std::string threshFile = "tree.dat";
        #ifdef _WIN32
        thresh_fd = ::_open(threshFile.c_str(), _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
        #else
        thresh_fd = ::open(threshFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        #endif
        if (thresh_fd < 0) {
            std::cerr << "Error creating thresholds file." << std::endl;
            std::cerr.flush();
            return false;
        }
        size_t threshSize = num_points * sizeof(NodeInfo);
        if (::ftruncate(thresh_fd, threshSize) < 0) {
            std::cerr << "Error truncating thresholds file." << std::endl;
            std::cerr.flush();
            return false;
        }
        nodeInfos = (NodeInfo*) mmap(nullptr, threshSize, PROT_READ | PROT_WRITE, MAP_SHARED, thresh_fd, 0);
        if (nodeInfos == MAP_FAILED) {
            std::cerr << "Error mmapping thresholds file." << std::endl;
            std::cerr.flush();
            return false;
        }
        // Allocate temporary distance buffer.
        std::cerr << "Number of points: " << num_points << std::endl;
        std::cerr.flush();
        float* buffer = new float[num_points];
        // Build the tree in place (reordering both embeddings and values) and fill nodeInfos.
        buildTreeInPlace(emb, vals, 0, num_points, dimension, nodeInfos, buffer);
        delete[] buffer;
        msync(emb, fileSize, MS_SYNC);
        msync(vals, valSize, MS_SYNC);
        msync(nodeInfos, threshSize, MS_SYNC);
        return true;
    }

    // Open an existing VP-tree given the embeddings file, values file, and tree file.
    bool open(const std::string& embedFile, const std::string& valueFile, const std::string& threshFile, int dim) {
        dimension = dim;
        // Open embeddings file.
        #ifdef _WIN32
        embed_fd = ::_open(embedFile.c_str(), _O_RDWR | _O_BINARY);
        #else
        embed_fd = ::open(embedFile.c_str(), O_RDWR);
        #endif
        if (embed_fd < 0) {
            std::cerr << "Error opening embeddings file: " << embedFile << "\n";
            return false;
        }
        struct stat st;
        if (fstat(embed_fd, &st) < 0) {
            std::cerr << "Error fstat on embeddings file.\n";
            return false;
        }
        size_t size = st.st_size;
        num_points = size / (dimension * sizeof(T));
        emb = (T*) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, embed_fd, 0);
        if (emb == MAP_FAILED) {
            std::cerr << "Error mmapping embeddings file.\n";
            return false;
        }
        // Open values file.
        #ifdef _WIN32
        val_fd = ::_open(valueFile.c_str(), _O_RDWR | _O_BINARY);
        #else
        val_fd = ::open(valueFile.c_str(), O_RDWR);
        #endif
        if (val_fd < 0) {
            std::cerr << "Error opening values file: " << valueFile << "\n";
            return false;
        }
        size_t vsize;
        {
            struct stat vst;
            if (fstat(val_fd, &vst) < 0) {
                std::cerr << "Error fstat on values file.\n";
                return false;
            }
            vsize = vst.st_size;
        }
        vals = (int32_t*) mmap(nullptr, vsize, PROT_READ | PROT_WRITE, MAP_SHARED, val_fd, 0);
        if (vals == MAP_FAILED) {
            std::cerr << "Error mmapping values file.\n";
            return false;
        }
        // Open thresholds file.
        #ifdef _WIN32
        thresh_fd = ::_open(threshFile.c_str(), _O_RDWR | _O_BINARY);
        #else
        thresh_fd = ::open(threshFile.c_str(), O_RDWR);
        #endif
        if (thresh_fd < 0) {
            std::cerr << "Error opening thresholds file: " << threshFile << "\n";
            return false;
        }
        size_t tsize = num_points * sizeof(NodeInfo);
        nodeInfos = (NodeInfo*) mmap(nullptr, tsize, PROT_READ | PROT_WRITE, MAP_SHARED, thresh_fd, 0);
        if (nodeInfos == MAP_FAILED) {
            std::cerr << "Error mmapping thresholds file.\n";
            return false;
        }
        return true;
    }

    // Close all memory mappings and file descriptors.
    void close() {
        if (emb && emb != MAP_FAILED) {
            size_t size = num_points * dimension * sizeof(T);
            munmap(emb, size);
            emb = nullptr;
        }
        if (embed_fd >= 0) {
            #ifdef _WIN32
            ::_close(embed_fd);
            #else
            ::close(embed_fd);
            #endif
            embed_fd = -1;
        }
        if (vals && vals != MAP_FAILED) {
            size_t vsize = num_points * sizeof(int32_t);
            munmap(vals, vsize);
            vals = nullptr;
        }
        if (val_fd >= 0) {
            #ifdef _WIN32
            ::_close(val_fd);
            #else
            ::close(val_fd);
            #endif
            val_fd = -1;
        }
        if (nodeInfos && nodeInfos != MAP_FAILED) {
            size_t tsize = num_points * sizeof(NodeInfo);
            munmap(nodeInfos, tsize);
            nodeInfos = nullptr;
        }
        if (thresh_fd >= 0) {
            #ifdef _WIN32
            ::_close(thresh_fd);
            #else
            ::close(thresh_fd);
            #endif
            thresh_fd = -1;
        }
    }

    // Search: find the nearest neighbor to the query vector (using Manhattan distance).
    // Returns the index (in the reordered embeddings file) of the best match.
    // (The corresponding value for that embedding can then be read from the values file.)
    int search(const std::vector<float>& query, float tau) {
        int bestIndex = -1;
        float bestDistance = std::numeric_limits<float>::infinity();
        searchRecursive(0, nodeInfos[0].totalSize, query, bestIndex, bestDistance, 1, tau);
        return vals[bestIndex];
    }

private:
    int embed_fd, val_fd, thresh_fd;
    T* emb;
    int32_t* vals;
    NodeInfo* nodeInfos;
    size_t num_points;
    int dimension;

    // Recursive search: process subtree at 'start' of size 'count'.
    void searchRecursive(int start, int count, const std::vector<float>& query, int &bestIndex, float &bestDistance, int depth, float tau) {
        if (count <= 0)
            return;
        float d = manhattanDistance(emb, start, query, dimension);
        if (d < bestDistance) {
            bestDistance = d;
            bestIndex = start;
        }
        float diff = d - nodeInfos[start].threshold;
        int leftCount = nodeInfos[start].leftSize;
        int total = nodeInfos[start].totalSize;
        // Left subtree is at [start+1, start+1+leftCount)
        // Right subtree is at [start+1+leftCount, start+total)
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
        if (argc != 6) {
            std::cerr << "Build mode requires 4 arguments: [emb_file] [val_file] [dim] [precision]\n";
            return 1;
        }
        std::string embFile = argv[2];
        std::string valFile = argv[3];
        int dim = std::stoi(argv[4]);
        
        if (precision == "float32") {
            VPTree<float> tree;
            if (!tree.build(embFile, valFile, dim)) {
                std::cerr << "Failed to build VP Tree in place (float32).\n";
                return 1;
            }
            tree.close();
        } else if (precision == "float16") {
            VPTree<Half> tree;
            if (!tree.build(embFile, valFile, dim)) {
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
            std::cerr << "Search mode requires " << (7 + dimVal) << " arguments.\n";
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
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("dim"))
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
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("dim"))
        .def("open", &VPTree<Half>::open,
             "Open existing VP tree (float16)",
             py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
        .def("close", &VPTree<Half>::close)
        .def("search", &VPTree<Half>::search,
             "Search VP tree (float16)", py::arg("query"), py::arg("tau"));
}
#endif






// // Compile:
// //  Bindings: python setup.py build_ext --inplace

// // Enable POSIX features.
// #define _POSIX_C_SOURCE 200809L

// #include <cstdint>
// #include <cstdlib>
// #include <ctime>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <string>
// #include <cmath>
// #include <fstream>
// #include <limits>
// #include <cstring>     // For memcpy

// // Platform-specific includes
// #ifdef _WIN32
//     #include <windows.h>
//     #include <io.h>
//     #include <fcntl.h>
    
//     // For stat compatibility
//     #define stat _stat64
//     #define fstat _fstat64
    
//     // Function to get file size in Windows
//     off_t fsize(int fd) {
//         struct _stat64 st;
//         if (_fstat64(fd, &st) < 0) {
//             return -1;
//         }
//         return st.st_size;
//     }
    
//     // Truncate function for Windows
//     int ftruncate(int fd, off_t length) {
//         HANDLE hFile = (HANDLE)_get_osfhandle(fd);
//         if (hFile == INVALID_HANDLE_VALUE) {
//             return -1;
//         }
        
//         LARGE_INTEGER li;
//         li.QuadPart = length;
        
//         if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) {
//             return -1;
//         }
        
//         if (!SetEndOfFile(hFile)) {
//             return -1;
//         }
        
//         return 0;
//     }
    
//     // Windows seek function
//     off_t lseek64(int fd, off_t offset, int whence) {
//         return _lseeki64(fd, offset, whence);
//     }
    
//     // Windows read/write functions
//     int read_file(int fd, void* buffer, size_t count) {
//         return _read(fd, buffer, static_cast<unsigned int>(count));
//     }
    
//     int write_file(int fd, const void* buffer, size_t count) {
//         return _write(fd, buffer, static_cast<unsigned int>(count));
//     }
    
// #else
//     #include <sys/stat.h>
//     #include <fcntl.h>
//     #include <unistd.h>
    
//     // UNIX seek function
//     off_t lseek64(int fd, off_t offset, int whence) {
//         return lseek(fd, offset, whence);
//     }
    
//     // UNIX read/write functions
//     ssize_t read_file(int fd, void* buffer, size_t count) {
//         return read(fd, buffer, count);
//     }
    
//     ssize_t write_file(int fd, const void* buffer, size_t count) {
//         return write(fd, buffer, count);
//     }
// #endif

// // Buffer sizes for file operations - adjust as needed
// #define BUFFER_SIZE 1048576  // 1MB buffer for reading/writing

// // Prototypes for half-precision conversion functions.
// uint16_t float_to_half(float f);
// float half_to_float(uint16_t h);

// // ---------------------------
// // Minimal half-precision type
// // ---------------------------
// struct Half {
//     uint16_t bits;
//     Half() : bits(0) {}
//     Half(float f) { bits = float_to_half(f); }
//     operator float() const { return half_to_float(bits); }
// };

// uint16_t float_to_half(float f) {
//     uint32_t x = *reinterpret_cast<uint32_t*>(&f);
//     uint16_t sign = (x >> 16) & 0x8000;
//     int exp = ((x >> 23) & 0xff) - 127 + 15;
//     if(exp <= 0)
//         return sign;
//     else if(exp >= 31)
//         return sign | 0x7c00;
//     else {
//         uint16_t mantissa = (x >> 13) & 0x3ff;
//         return sign | (exp << 10) | mantissa;
//     }
// }

// float half_to_float(uint16_t h) {
//     uint16_t sign = h & 0x8000;
//     uint16_t exp = (h >> 10) & 0x1f;
//     uint16_t frac = h & 0x3ff;
//     uint32_t f;
//     if(exp == 0) {
//         if(frac == 0) {
//             f = sign << 16;
//         } else {
//             exp = 1;
//             while((frac & 0x400) == 0) {
//                 frac <<= 1;
//                 exp--;
//             }
//             frac &= 0x3ff;
//             f = (sign << 16) | ((exp - 1 + 127) << 23) | (frac << 13);
//         }
//     } else if(exp == 31) {
//         f = (sign << 16) | 0x7f800000 | (frac << 13);
//     } else {
//         f = (sign << 16) | ((exp - 15 + 127) << 23) | (frac << 13);
//     }
//     return *reinterpret_cast<float*>(&f);
// }

// // ------------------------------
// // NodeInfo: stored per node in tree file.
// // We keep leftSize and totalSize to quickly compute subtree boundaries during search.
// struct NodeInfo {
//     float threshold;
//     int leftSize;    // number of nodes in left subtree
//     int totalSize;   // total nodes in this subtree (including current node)
// };

// // ------------------------------
// // File IO Helper functions

// // Read an embedding of type T at a specific index
// template <typename T>
// void readEmbedding(int fd, int index, int dim, T* buffer) {
//     off_t offset = static_cast<off_t>(index) * dim * sizeof(T);
//     lseek64(fd, offset, SEEK_SET);
//     read_file(fd, buffer, dim * sizeof(T));
// }

// // Write an embedding of type T at a specific index
// template <typename T>
// void writeEmbedding(int fd, int index, int dim, const T* buffer) {
//     off_t offset = static_cast<off_t>(index) * dim * sizeof(T);
//     lseek64(fd, offset, SEEK_SET);
//     write_file(fd, buffer, dim * sizeof(T));
// }

// // Read a value at a specific index
// void readValue(int fd, int index, int32_t* value) {
//     off_t offset = static_cast<off_t>(index) * sizeof(int32_t);
//     lseek64(fd, offset, SEEK_SET);
//     read_file(fd, value, sizeof(int32_t));
// }

// // Write a value at a specific index
// void writeValue(int fd, int index, const int32_t* value) {
//     off_t offset = static_cast<off_t>(index) * sizeof(int32_t);
//     lseek64(fd, offset, SEEK_SET);
//     write_file(fd, value, sizeof(int32_t));
// }

// // Read a node info at a specific index
// void readNodeInfo(int fd, int index, NodeInfo* node) {
//     off_t offset = static_cast<off_t>(index) * sizeof(NodeInfo);
//     lseek64(fd, offset, SEEK_SET);
//     read_file(fd, node, sizeof(NodeInfo));
// }

// // Write a node info at a specific index
// void writeNodeInfo(int fd, int index, const NodeInfo* node) {
//     off_t offset = static_cast<off_t>(index) * sizeof(NodeInfo);
//     lseek64(fd, offset, SEEK_SET);
//     write_file(fd, node, sizeof(NodeInfo));
// }

// // Swap embeddings and values at two indices
// template <typename T>
// void swapEmbAndVal(int embed_fd, int val_fd, int idx1, int idx2, int dim, T* embBuffer1, T* embBuffer2, int32_t* valBuffer1, int32_t* valBuffer2) {
//     // Read the embeddings and values
//     readEmbedding(embed_fd, idx1, dim, embBuffer1);
//     readEmbedding(embed_fd, idx2, dim, embBuffer2);
//     readValue(val_fd, idx1, valBuffer1);
//     readValue(val_fd, idx2, valBuffer2);
    
//     // Write them back swapped
//     writeEmbedding(embed_fd, idx1, dim, embBuffer2);
//     writeEmbedding(embed_fd, idx2, dim, embBuffer1);
//     writeValue(val_fd, idx1, valBuffer2);
//     writeValue(val_fd, idx2, valBuffer1);
// }

// // Helper: compute Manhattan distance between two embeddings
// template <typename T>
// float manhattanDistance(const T* emb1, const T* emb2, int dim) {
//     float dist = 0;
//     for (int i = 0; i < dim; i++) {
//         float a = static_cast<float>(emb1[i]);
//         float b = static_cast<float>(emb2[i]);
//         dist += std::fabs(a - b);
//     }
//     return dist;
// }

// // Helper: compute Manhattan distance between an embedding in a file and another embedding in memory
// template <typename T>
// float manhattanDistance(int embed_fd, int idx, T* embBuffer, const T* query, int dim) {
//     readEmbedding(embed_fd, idx, dim, embBuffer);
    
//     float dist = 0;
//     for (int i = 0; i < dim; i++) {
//         float a = static_cast<float>(embBuffer[i]);
//         float b = static_cast<float>(query[i]);
//         dist += std::fabs(a - b);
//     }
//     return dist;
// }

// // Helper: compute Manhattan distance between an embedding in a file and a query vector
// template <typename T>
// float manhattanDistance(int embed_fd, int idx, T* embBuffer, const std::vector<float>& query, int dim) {
//     readEmbedding(embed_fd, idx, dim, embBuffer);
    
//     float dist = 0;
//     for (int i = 0; i < dim; i++) {
//         float a = static_cast<float>(embBuffer[i]);
//         dist += std::fabs(a - query[i]);
//     }
//     return dist;
// }

// // ------------------------------
// // Recursive function that builds the VP-tree by reordering the embeddings
// // and the corresponding values (in the range [start, start+count))
// template <typename T>
// int buildTreeInPlace(int embed_fd, int val_fd, int thresh_fd, int start, int count, int dim, 
//                     T* vpBuffer, T* embBuffer, int32_t* valBuffer1, int32_t* valBuffer2, float* distBuffer) {
//     if (count <= 0)
//         return 0;
    
//     NodeInfo nodeInfo;
    
//     if (count == 1) {
//         nodeInfo.threshold = 0;
//         nodeInfo.leftSize = 0;
//         nodeInfo.totalSize = 1;
//         writeNodeInfo(thresh_fd, start, &nodeInfo);
//         return 1;
//     }
    
//     // Choose a random vantage point from current segment
//     int pivot = start + (std::rand() % count);
    
//     // Swap chosen vantage point to start position
//     swapEmbAndVal(embed_fd, val_fd, pivot, start, dim, vpBuffer, embBuffer, valBuffer1, valBuffer2);
    
//     int vpIndex = start;
//     // Read vantage point embedding
//     readEmbedding(embed_fd, vpIndex, dim, vpBuffer);
    
//     // Compute distances from each embedding (except vp) to the vantage point
//     for (int i = start + 1; i < start + count; i++) {
//         readEmbedding(embed_fd, i, dim, embBuffer);
//         distBuffer[i - start - 1] = manhattanDistance(vpBuffer, embBuffer, dim);
//     }
    
//     // Find median of these distances
//     int m = (count - 1) / 2;
//     std::nth_element(distBuffer, distBuffer + m, distBuffer + count - 1);
//     float median = distBuffer[m];
    
//     // Partition embeddings in [start+1, start+count) by median
//     int i = start + 1, j = start + count - 1;
//     float swapBuff = 0;
    
//     while (i <= j) {
//         // float d;
//         // readEmbedding(embed_fd, i, dim, embBuffer);
//         // d = manhattanDistance(vpBuffer, embBuffer, dim);
        
//         float d = distBuffer[i - start - 1];
//         if (d <= median) {
//             i++;
//         } else {
//             swapBuff = distBuffer[i - start - 1];
//             distBuffer[i - start - 1] = distBuffer[j - start];
//             distBuffer[j - start] = swapBuff;
//             swapEmbAndVal(embed_fd, val_fd, i, j, dim, vpBuffer, embBuffer, valBuffer1, valBuffer2);
//             j--;
//         }
//     }
    
//     int leftCount = i - start - 1;
    
//     // Recurse: left subtree in [start+1, start+1+leftCount), right subtree in remaining segment
//     int leftSize = buildTreeInPlace(embed_fd, val_fd, thresh_fd, start + 1, leftCount, dim, 
//                                   vpBuffer, embBuffer, valBuffer1, valBuffer2, distBuffer);
//     int rightSize = buildTreeInPlace(embed_fd, val_fd, thresh_fd, start + 1 + leftCount, count - 1 - leftCount, dim, 
//                                    vpBuffer, embBuffer, valBuffer1, valBuffer2, distBuffer);
    
//     int totalSize = 1 + leftSize + rightSize;
    
//     nodeInfo.threshold = median;
//     nodeInfo.leftSize = leftSize;
//     nodeInfo.totalSize = totalSize;
//     writeNodeInfo(thresh_fd, start, &nodeInfo);
    
//     return totalSize;
// }

// // ------------------------------
// // VP-Tree class that uses file IO instead of memory mapping for large files
// template <typename T>
// class VPTree {
// public:
//     VPTree() : embed_fd(-1), val_fd(-1), thresh_fd(-1), num_points(0), dimension(0) {
//         std::srand(std::time(nullptr));
//     }
    
//     ~VPTree() { close(); }
    
//     // Build: Reorder the embeddings file and the corresponding values file,
//     // and write node info into "tree.dat".
//     bool build(const std::string& embedFile, const std::string& valueFile, int dim) {
//         dimension = dim;
        
//         // Open embeddings file (read-write)
//         #ifdef _WIN32
//         embed_fd = ::_open(embedFile.c_str(), _O_RDWR | _O_BINARY);
//         #else
//         embed_fd = ::open(embedFile.c_str(), O_RDWR);
//         #endif
        
//         if (embed_fd < 0) {
//             std::cerr << "Error opening embeddings file: " << embedFile << "\n";
//             return false;
//         }
        
//         struct stat st;
//         if (fstat(embed_fd, &st) < 0) {
//             std::cerr << "Error fstat on embeddings file.\n";
//             return false;
//         }
        
//         size_t fileSize = st.st_size;
//         num_points = fileSize / (dimension * sizeof(T));
        
//         // Open values file (read-write)
//         #ifdef _WIN32
//         val_fd = ::_open(valueFile.c_str(), _O_RDWR | _O_BINARY);
//         #else
//         val_fd = ::open(valueFile.c_str(), O_RDWR);
//         #endif
        
//         if (val_fd < 0) {
//             std::cerr << "Error opening values file: " << valueFile << "\n";
//             return false;
//         }
        
//         struct stat vst;
//         if (fstat(val_fd, &vst) < 0) {
//             std::cerr << "Error fstat on values file.\n";
//             return false;
//         }
        
//         size_t valSize = vst.st_size;
//         if (valSize != num_points * sizeof(int32_t)) {
//             std::cerr << "Mismatch between embeddings and values count.\n";
//             return false;
//         }
        
//         // Create thresholds file ("tree.dat")
//         std::string threshFile = "tree.dat";
//         #ifdef _WIN32
//         thresh_fd = ::_open(threshFile.c_str(), _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
//         #else
//         thresh_fd = ::open(threshFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
//         #endif
        
//         if (thresh_fd < 0) {
//             std::cerr << "Error creating thresholds file.\n";
//             return false;
//         }
        
//         size_t threshSize = num_points * sizeof(NodeInfo);
//         if (::ftruncate(thresh_fd, threshSize) < 0) {
//             std::cerr << "Error truncating thresholds file.\n";
//             return false;
//         }
        
//         // Allocate buffers for reading/writing
//         T* vpBuffer = new T[dimension];        // Buffer for vantage point
//         T* embBuffer = new T[dimension];       // Buffer for embeddings
//         int32_t* valBuffer1 = new int32_t[1];  // Buffer for values
//         int32_t* valBuffer2 = new int32_t[1];  // Second buffer for swapping
//         float* distBuffer = new float[num_points]; // Buffer for distances
        
//         std::cout << "Number of points: " << num_points << "\n";
        
//         // Build the tree
//         buildTreeInPlace(embed_fd, val_fd, thresh_fd, 0, num_points, dimension, 
//                         vpBuffer, embBuffer, valBuffer1, valBuffer2, distBuffer);
        
//         // Free buffers
//         delete[] vpBuffer;
//         delete[] embBuffer;
//         delete[] valBuffer1;
//         delete[] valBuffer2;
//         delete[] distBuffer;
        
//         return true;
//     }
    
//     // Open an existing VP-tree
//     bool open(const std::string& embedFile, const std::string& valueFile, const std::string& threshFile, int dim) {
//         dimension = dim;
        
//         // Open embeddings file
//         #ifdef _WIN32
//         embed_fd = ::_open(embedFile.c_str(), _O_RDONLY | _O_BINARY);
//         #else
//         embed_fd = ::open(embedFile.c_str(), O_RDONLY);
//         #endif
        
//         if (embed_fd < 0) {
//             std::cerr << "Error opening embeddings file: " << embedFile << "\n";
//             return false;
//         }
        
//         struct stat st;
//         if (fstat(embed_fd, &st) < 0) {
//             std::cerr << "Error fstat on embeddings file.\n";
//             return false;
//         }
        
//         size_t size = st.st_size;
//         num_points = size / (dimension * sizeof(T));
        
//         // Open values file
//         #ifdef _WIN32
//         val_fd = ::_open(valueFile.c_str(), _O_RDONLY | _O_BINARY);
//         #else
//         val_fd = ::open(valueFile.c_str(), O_RDONLY);
//         #endif
        
//         if (val_fd < 0) {
//             std::cerr << "Error opening values file: " << valueFile << "\n";
//             return false;
//         }
        
//         // Open thresholds file
//         #ifdef _WIN32
//         thresh_fd = ::_open(threshFile.c_str(), _O_RDONLY | _O_BINARY);
//         #else
//         thresh_fd = ::open(threshFile.c_str(), O_RDONLY);
//         #endif
        
//         if (thresh_fd < 0) {
//             std::cerr << "Error opening thresholds file: " << threshFile << "\n";
//             return false;
//         }
        
//         return true;
//     }
    
//     // Close all file descriptors
//     void close() {
//         if (embed_fd >= 0) {
//             #ifdef _WIN32
//             ::_close(embed_fd);
//             #else
//             ::close(embed_fd);
//             #endif
//             embed_fd = -1;
//         }
        
//         if (val_fd >= 0) {
//             #ifdef _WIN32
//             ::_close(val_fd);
//             #else
//             ::close(val_fd);
//             #endif
//             val_fd = -1;
//         }
        
//         if (thresh_fd >= 0) {
//             #ifdef _WIN32
//             ::_close(thresh_fd);
//             #else
//             ::close(thresh_fd);
//             #endif
//             thresh_fd = -1;
//         }
//     }
    
//     // Search: find the nearest neighbor to the query vector (using Manhattan distance)
//     int search(const std::vector<float>& query, float tau) {
//         int bestIndex = -1;
//         float bestDistance = std::numeric_limits<float>::infinity();
        
//         // Allocate a buffer for reading embeddings
//         T* embBuffer = new T[dimension];
        
//         // Read the root node info
//         NodeInfo rootNode;
//         readNodeInfo(thresh_fd, 0, &rootNode);
        
//         // Search recursively
//         searchRecursive(0, rootNode.totalSize, query, bestIndex, bestDistance, 1, tau, embBuffer);
        
//         // Get the actual value
//         int32_t result = -1;
//         if (bestIndex != -1) {
//             int32_t* valBuffer = new int32_t[1];
//             readValue(val_fd, bestIndex, valBuffer);
//             result = *valBuffer;
//             delete[] valBuffer;
//         }
        
//         delete[] embBuffer;
//         return result;
//     }
    
// private:
//     int embed_fd, val_fd, thresh_fd;
//     size_t num_points;
//     int dimension;
    
//     // Recursive search: process subtree at 'start' of size 'count'
//     void searchRecursive(int start, int count, const std::vector<float>& query, 
//                          int &bestIndex, float &bestDistance, int depth, float tau, T* embBuffer) {
//         if (count <= 0)
//             return;
        
//         // Read current node info
//         NodeInfo nodeInfo;
//         readNodeInfo(thresh_fd, start, &nodeInfo);
        
//         // Calculate distance to vantage point
//         float d = manhattanDistance(embed_fd, start, embBuffer, query, dimension);
        
//         if (d < bestDistance) {
//             bestDistance = d;
//             bestIndex = start;
//         }
        
//         float diff = d - nodeInfo.threshold;
//         int leftCount = nodeInfo.leftSize;
//         int total = nodeInfo.totalSize;
        
//         // Left subtree is at [start+1, start+1+leftCount)
//         // Right subtree is at [start+1+leftCount, start+total)
//         if (diff < 0) {
//             searchRecursive(start + 1, leftCount, query, bestIndex, bestDistance, depth+1, tau, embBuffer);
//             float prob = std::max<float>(1.0f, std::log2((float)depth)) * tau;
//             if (std::fabs(diff) < bestDistance * prob)
//                 searchRecursive(start + 1 + leftCount, total - 1 - leftCount, query, bestIndex, bestDistance, depth+1, tau, embBuffer);
//         } else {
//             searchRecursive(start + 1 + leftCount, total - 1 - leftCount, query, bestIndex, bestDistance, depth+1, tau, embBuffer);
//             float prob = std::max<float>(1.0f, std::log2((float)depth)) * tau;
//             if (std::fabs(diff) < bestDistance * prob)
//                 searchRecursive(start + 1, leftCount, query, bestIndex, bestDistance, depth+1, tau, embBuffer);
//         }
//     }
// };

// #ifndef BUILD_PYBINDINGS

// int main(int argc, char** argv) {
//     // Usage:
//     // Build:   build [emb_file] [val_file] [dim] [precision]
//     // Search:  search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]
//     if (argc < 6) {
//         std::cout << "Usage:\n"
//                   << "  Build:  " << argv[0] << " build [emb_file] [val_file] [dim] [precision]\n"
//                   << "  Search: " << argv[0] << " search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]\n";
//         return 1;
//     }
    
//     std::string mode = argv[1];
//     // The last argument is used to specify the precision.
//     std::string precision = argv[argc - 1];
    
//     if (mode == "build") {
//         if (argc != 6) {
//             std::cerr << "Build mode requires 4 arguments: [emb_file] [val_file] [dim] [precision]\n";
//             return 1;
//         }
//         std::string embFile = argv[2];
//         std::string valFile = argv[3];
//         int dim = std::stoi(argv[4]);
        
//         if (precision == "float32") {
//             VPTree<float> tree;
//             if (!tree.build(embFile, valFile, dim)) {
//                 std::cerr << "Failed to build VP Tree in place (float32).\n";
//                 return 1;
//             }
//             tree.close();
//         } else if (precision == "float16") {
//             VPTree<Half> tree;
//             if (!tree.build(embFile, valFile, dim)) {
//                 std::cerr << "Failed to build VP Tree in place (float16).\n";
//                 return 1;
//             }
//             tree.close();
//         } else {
//             std::cerr << "Unknown precision: " << precision << "\n";
//             return 1;
//         }
//         std::cout << "VP Tree built successfully.\n";
        
//     } else if (mode == "search") {
//         // Expected arguments:
//         // search [emb_file] [val_file] [tree_file] [dim] [query_vector...] [tau] [precision]
//         // Total arguments = 1(mode) + 3(files) + 1(dim) + (dim query elements) + 1(tau) + 1(precision)
//         int dimVal = std::stoi(argv[5]); // argv[5] holds the dimension.
//         int expectedArgs = 8 + dimVal; // 1+1+3+1+dim+1+1
//         if (argc != expectedArgs) {
//             std::cerr << "Search mode requires " << (7 + dimVal) << " arguments.\n";
//             return 1;
//         }
//         std::string embFile = argv[2];
//         std::string valFile = argv[3];
//         std::string treeFile = argv[4];
//         int dim = dimVal;
//         std::vector<float> query(dim);
//         for (int i = 0; i < dim; i++) {
//             query[i] = std::stof(argv[6 + i]); // Query elements from argv[6] to argv[6+dim-1]
//         }
//         float tau = std::stof(argv[6 + dim]); // Tau comes after the query elements.
        
//         if (precision == "float32") {
//             VPTree<float> tree;
//             if (!tree.open(embFile, valFile, treeFile, dim)) {
//                 std::cerr << "Failed to open VP Tree (float32).\n";
//                 return 1;
//             }
//             int best = tree.search(query, tau);
//             if (best == -1)
//                 std::cout << "No neighbor found within tau.\n";
//             else
//                 std::cout << "Nearest neighbor (float32) value: " << best << "\n";
//             tree.close();
//         } else if (precision == "float16") {
//             VPTree<Half> tree;
//             if (!tree.open(embFile, valFile, treeFile, dim)) {
//                 std::cerr << "Failed to open VP Tree (float16).\n";
//                 return 1;
//             }
//             int best = tree.search(query, tau);
//             if (best == -1)
//                 std::cout << "No neighbor found within tau.\n";
//             else
//                 std::cout << "Nearest neighbor (float16) value: " << best << "\n";
//             tree.close();
//         } else {
//             std::cerr << "Unknown precision: " << precision << "\n";
//             return 1;
//         }
//     } else {
//         std::cerr << "Unknown mode: " << mode << "\n";
//         return 1;
//     }
//     return 0;
// }

// #endif // BUILD_PYBINDINGS

// #ifdef BUILD_PYBINDINGS
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// namespace py = pybind11;

// // In Python, you choose the proper class based on your data precision.
// PYBIND11_MODULE(diskvec, m) {
//     m.doc() = "VP Tree module with file-based I/O for handling large embedding files. "
//               "Use VPTreeFloat for float32 data and VPTreeHalf for float16 data.";
    
//     py::class_<VPTree<float>>(m, "VPTreeFloat")
//         .def(py::init<>())
//         .def("build", &VPTree<float>::build,
//              "Build VP tree in place from embeddings and values files (float32)",
//              py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("dim"))
//         .def("open", &VPTree<float>::open,
//              "Open existing VP tree (float32)",
//              py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
//         .def("close", &VPTree<float>::close)
//         .def("search", &VPTree<float>::search,
//              "Search VP tree (float32)", py::arg("query"), py::arg("tau"));
    
//     py::class_<VPTree<Half>>(m, "VPTreeHalf")
//         .def(py::init<>())
//         .def("build", &VPTree<Half>::build,
//              "Build VP tree in place from embeddings and values files (float16)",
//              py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("dim"))
//         .def("open", &VPTree<Half>::open,
//              "Open existing VP tree (float16)",
//              py::arg("embeddingsFile"), py::arg("valuesFile"), py::arg("treeFile"), py::arg("dim"))
//         .def("close", &VPTree<Half>::close)
//         .def("search", &VPTree<Half>::search,
//              "Search VP tree (float16)", py::arg("query"), py::arg("tau"));
// }
// #endif