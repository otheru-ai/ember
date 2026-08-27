#include "qwen4exp_activation_dump.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace dflash::common {
namespace {

constexpr size_t kRecordBytes = kQwen4ExpActivationFloats * sizeof(float);

class FileDescriptor {
public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor() { if (value_ >= 0) ::close(value_); }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor & operator=(const FileDescriptor &) = delete;
    int get() const { return value_; }
    int release() { const int value = value_; value_ = -1; return value; }

private:
    int value_;
};

std::string system_error(const char * operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

bool write_all(int fd, const void * data, size_t bytes, std::string & error) {
    const auto * cursor = static_cast<const unsigned char *>(data);
    while (bytes > 0) {
        const ssize_t written = ::write(fd, cursor, bytes);
        if (written < 0) {
            if (errno == EINTR) continue;
            error = system_error("write activation dump");
            return false;
        }
        if (written == 0) {
            error = "write activation dump returned zero bytes";
            return false;
        }
        const size_t count = static_cast<size_t>(written);
        cursor += count;
        bytes -= count;
    }
    return true;
}

bool copy_all(int source, int destination, std::string & error) {
    unsigned char buffer[1U << 20];
    for (;;) {
        const ssize_t got = ::read(source, buffer, sizeof(buffer));
        if (got < 0) {
            if (errno == EINTR) continue;
            error = system_error("read existing activation dump");
            return false;
        }
        if (got == 0) return true;
        if (!write_all(destination, buffer, static_cast<size_t>(got), error))
            return false;
    }
}

bool little_endian_f32_host() {
    const uint32_t value = 0x3f800000U;
    const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
    return sizeof(float) == 4 && bytes[0] == 0x00U && bytes[1] == 0x00U &&
           bytes[2] == 0x80U && bytes[3] == 0x3fU;
}

std::string parent_directory(const std::string & path) {
    const size_t slash = path.rfind('/');
    return slash == 0 ? "/" : path.substr(0, slash);
}

} // namespace

bool qwen4exp_append_activation_dump(
    const std::string & path, const std::vector<float> & activations,
    Qwen4ExpActivationDumpResult & result, std::string & error) {
    result = {};
    error.clear();
    if (path.empty() || path.front() != '/' || path.back() == '/') {
        error = "Qwen4Exp activation dump path must be an absolute file path";
        return false;
    }
    if (!little_endian_f32_host()) {
        error = "Qwen4Exp activation dumps require little-endian IEEE-754 F32";
        return false;
    }
    if (activations.size() != kQwen4ExpActivationFloats) {
        error = "Qwen4Exp activation record must contain exactly 48x2560 floats";
        return false;
    }
    for (float value : activations) {
        if (!std::isfinite(value)) {
            error = "Qwen4Exp activation record contains a non-finite value";
            return false;
        }
    }

    const std::string lock_path = path + ".lock";
    FileDescriptor lock(::open(lock_path.c_str(),
                               O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                               S_IRUSR | S_IWUSR));
    if (lock.get() < 0) {
        error = system_error("open activation dump lock");
        return false;
    }
    if (::flock(lock.get(), LOCK_EX) != 0) {
        error = system_error("lock activation dump");
        return false;
    }

    uint64_t existing_bytes = 0;
    FileDescriptor existing(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (existing.get() < 0 && errno != ENOENT) {
        error = system_error("open existing activation dump");
        return false;
    }
    if (existing.get() >= 0) {
        struct stat status {};
        if (::fstat(existing.get(), &status) != 0) {
            error = system_error("stat existing activation dump");
            return false;
        }
        if (!S_ISREG(status.st_mode) || status.st_size < 0) {
            error = "existing Qwen4Exp activation dump is not a regular file";
            return false;
        }
        existing_bytes = static_cast<uint64_t>(status.st_size);
        if (existing_bytes % kRecordBytes != 0) {
            error = "existing Qwen4Exp activation dump ends in a partial record";
            return false;
        }
    }
    if (existing_bytes > std::numeric_limits<uint64_t>::max() - kRecordBytes) {
        error = "Qwen4Exp activation dump size overflows uint64";
        return false;
    }

    std::string temporary = path + ".tmp.XXXXXX";
    std::vector<char> name(temporary.begin(), temporary.end());
    name.push_back('\0');
    FileDescriptor output(::mkstemp(name.data()));
    if (output.get() < 0) {
        error = system_error("create activation dump transaction");
        return false;
    }
    const std::string temporary_path(name.data());
    bool keep_temporary = true;
    auto rollback = [&]() {
        if (keep_temporary) (void)::unlink(temporary_path.c_str());
    };
    if (::fchmod(output.get(), S_IRUSR | S_IWUSR) != 0 ||
        (existing.get() >= 0 && !copy_all(existing.get(), output.get(), error)) ||
        !write_all(output.get(), activations.data(), kRecordBytes, error) ||
        ::fdatasync(output.get()) != 0) {
        if (error.empty()) error = system_error("sync activation dump transaction");
        rollback();
        return false;
    }
    const int raw_output = output.release();
    if (::close(raw_output) != 0) {
        error = system_error("close activation dump transaction");
        rollback();
        return false;
    }
    if (::rename(temporary_path.c_str(), path.c_str()) != 0) {
        error = system_error("commit activation dump transaction");
        rollback();
        return false;
    }
    keep_temporary = false;

    FileDescriptor directory(::open(parent_directory(path).c_str(),
                                    O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.get() < 0 || ::fsync(directory.get()) != 0) {
        // The file was renamed successfully and is internally complete. Report
        // the durability failure so the extraction runner does not count it as
        // certified evidence after a host crash.
        error = directory.get() < 0
            ? system_error("open activation dump directory")
            : system_error("sync activation dump directory");
        return false;
    }

    result.ordinal = existing_bytes / kRecordBytes;
    result.byte_offset = existing_bytes;
    return true;
}

} // namespace dflash::common
