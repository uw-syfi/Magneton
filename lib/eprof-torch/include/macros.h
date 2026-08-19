#pragma once

// EPROF_CHECK: abort with a message. Unconditional -- it stays in release
// builds, because what it guards are invariants of the capture cycle rather
// than debug aids, and continuing past a broken one produces a plausible-
// looking trace that is wrong.

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

inline auto EprofFormatMessage(const char* file, int line, const std::string& cond_str, const std::string& msg) -> std::string {
    std::ostringstream oss;
    oss << "[EPROF_CHECK FAILED] " << file << ":" << line << "\n"
        << "Condition: " << cond_str << "\n"
        << "Message: " << msg << "\n";
    return oss.str();
}

#define EPROF_CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            std::ostringstream __eprof_ss__; \
            __eprof_ss__ << __VA_ARGS__; \
            std::cerr << EprofFormatMessage(__FILE__, __LINE__, #cond, __eprof_ss__.str()) << std::endl; \
            std::abort(); \
        } \
    } while (0)
