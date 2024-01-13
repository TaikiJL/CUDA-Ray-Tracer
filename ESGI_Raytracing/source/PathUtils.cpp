#include "PathUtils.h"

#include <iostream>
#include <Windows.h>

std::string PathUtils::MakeAbsolutePath(const std::string& relativePath)
{
    char buffer[MAX_PATH];
    const DWORD length = GetModuleFileName(nullptr, buffer, MAX_PATH);

    if (length == 0) {
        // Handle error (print an error message, throw an exception, etc.)
        std::cerr << "Error getting the module filename." << std::endl;
        return nullptr;
    }

    // Convert to std::string
    std::string currentPath(buffer, length);

    size_t lastSlashPos = currentPath.find_last_of("\\/");
    lastSlashPos += sizeof(char);

    // TODO: check if buffer is big enough for the relative path
    std::strcpy(buffer + lastSlashPos, relativePath.c_str());

    std::string absolutePath(buffer, lastSlashPos + relativePath.length());

    return absolutePath;
}
