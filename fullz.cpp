#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <algorithm>

// 把正斜杠转反斜杠
std::string normalizePath(const std::string& path) {
    std::string result = path;
    std::replace(result.begin(), result.end(), '/', '\\');
    return result;
}

// 展开为完整路径
std::string getFullPath(const std::string& path) {
    char fullPath[MAX_PATH] = {0};
    if (GetFullPathNameA(path.c_str(), MAX_PATH, fullPath, nullptr)) {
        return fullPath;
    }
    return path;
}

// 获取驱动器根，如 "C:\\" 或 "Z:\\"
std::string getDriveRoot(const std::string& fullPath) {
    if (fullPath.size() >= 2 && fullPath[1] == ':') {
        return fullPath.substr(0, 2) + "\\";
    }
    return "";
}

// 转换 WebDAV UNC 路径为 HTTPS URL
std::string convertWebDAVToURL(const std::string& uncPath) {
    // 格式: \\host@SSL@port\DavWWWRoot\path -> https://host:port/path
    if (uncPath.find("@SSL@") != std::string::npos && uncPath.find("\\DavWWWRoot\\") != std::string::npos) {
        size_t start = 2; // 跳过 "\\"
        size_t sslPos = uncPath.find("@SSL@", start);
        if (sslPos == std::string::npos) return uncPath;
        
        std::string host = uncPath.substr(start, sslPos - start);
        size_t portStart = sslPos + 5; // 跳过 "@SSL@"
        size_t davPos = uncPath.find("\\DavWWWRoot\\", portStart);
        if (davPos == std::string::npos) return uncPath;
        
        std::string port = uncPath.substr(portStart, davPos - portStart);
        std::string path = uncPath.substr(davPos + 12); // 跳过 "\DavWWWRoot\"
        
        // 转换路径分隔符
        std::replace(path.begin(), path.end(), '\\', '/');
        
        return "https://" + host + ":" + port + "/" + path;
    }
    return uncPath;
}

// 通过 WNetGetConnection 查找 UNC 路径
std::string getUNCPath(const std::string& driveRoot) {
    char unc[MAX_PATH] = {0};
    DWORD size = MAX_PATH;
    // WNetGetConnectionA 需要 "X:" 格式，不是 "X:\"
    std::string driveLetter = driveRoot.substr(0, 2);
    DWORD result = WNetGetConnectionA(driveLetter.c_str(), unc, &size);
    if (result == NO_ERROR) {
        return unc;
    }
    return "";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: No parameters provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [file or folder] ..." << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        std::string item = normalizePath(argv[i]);

        // 检查路径是否存在
        DWORD attrs = GetFileAttributesA(item.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES) {
            std::cerr << "[ERROR] Path not found: " << item << std::endl;
            continue;
        }

        std::string fullPath = getFullPath(item);
        std::string driveRoot = getDriveRoot(fullPath);

        if (!driveRoot.empty()) {
            std::string unc = getUNCPath(driveRoot);
            if (!unc.empty()) {
                // 去掉盘符 "X:"，拼接 UNC 前缀
                std::string relative = fullPath.substr(2);
                std::string result = unc + relative;
                // 转换 WebDAV UNC 为 URL
                result = convertWebDAVToURL(result);
                std::cout << result << std::endl;
            } else {
                std::cout << fullPath << std::endl;
            }
        } else {
            std::cout << fullPath << std::endl;
        }
    }
    return 0;
}

