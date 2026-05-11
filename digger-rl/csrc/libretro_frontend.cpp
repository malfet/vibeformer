// Minimal libretro frontend exposed to Python via pybind11.
// Wraps a single libretro core (loaded via dlopen) as a Python class with
// run / load_game / get_frame / get_memory hooks. Singleton: only one core
// can be active per process because libretro callbacks have no userdata.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "libretro.h"
}

namespace py = pybind11;

class LibretroCore {
public:
    LibretroCore(const std::string& core_path,
                 const std::string& system_dir,
                 const std::string& save_dir)
        : system_dir_(system_dir), save_dir_(save_dir) {
        if (instance_) {
            throw std::runtime_error(
                "only one LibretroCore can be active per process");
        }
        instance_ = this;

        handle_ = dlopen(core_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }

#define RESOLVE(name) \
    name##_ = reinterpret_cast<decltype(name##_)>(dlsym(handle_, #name)); \
    if (!name##_) throw std::runtime_error("missing symbol: " #name)

        RESOLVE(retro_init);
        RESOLVE(retro_deinit);
        RESOLVE(retro_run);
        RESOLVE(retro_reset);
        RESOLVE(retro_load_game);
        RESOLVE(retro_unload_game);
        RESOLVE(retro_get_system_info);
        RESOLVE(retro_get_system_av_info);
        RESOLVE(retro_get_memory_data);
        RESOLVE(retro_get_memory_size);
        RESOLVE(retro_set_environment);
        RESOLVE(retro_set_video_refresh);
        RESOLVE(retro_set_audio_sample);
        RESOLVE(retro_set_audio_sample_batch);
        RESOLVE(retro_set_input_poll);
        RESOLVE(retro_set_input_state);
#undef RESOLVE

        retro_set_environment_(&LibretroCore::env_cb);
        retro_set_video_refresh_(&LibretroCore::video_cb);
        retro_set_audio_sample_(&LibretroCore::audio_sample_cb);
        retro_set_audio_sample_batch_(&LibretroCore::audio_batch_cb);
        retro_set_input_poll_(&LibretroCore::input_poll_cb);
        retro_set_input_state_(&LibretroCore::input_state_cb);

        retro_init_();
    }

    ~LibretroCore() {
        if (game_loaded_ && retro_unload_game_) retro_unload_game_();
        if (handle_) {
            if (retro_deinit_) retro_deinit_();
            dlclose(handle_);
        }
        instance_ = nullptr;
    }

    void load_game(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("can't open game: " + path);
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        game_data_.resize(sz);
        if (std::fread(game_data_.data(), 1, sz, f) != static_cast<size_t>(sz)) {
            std::fclose(f);
            throw std::runtime_error("short read on game file");
        }
        std::fclose(f);

        retro_game_info gi = {};
        gi.path = path.c_str();
        gi.data = game_data_.data();
        gi.size = static_cast<size_t>(sz);
        gi.meta = nullptr;

        if (!retro_load_game_(&gi)) {
            throw std::runtime_error("retro_load_game failed");
        }
        game_loaded_ = true;
    }

    void run() { retro_run_(); }
    void reset() { retro_reset_(); }

    py::array get_frame() {
        if (frame_w_ == 0 || frame_h_ == 0) {
            throw std::runtime_error("no frame available yet — call run() first");
        }
        // Return a copy as numpy (H, W, 4) uint8 in RGBA order.
        py::array_t<uint8_t> arr({static_cast<py::ssize_t>(frame_h_),
                                  static_cast<py::ssize_t>(frame_w_),
                                  static_cast<py::ssize_t>(4)});
        std::memcpy(arr.mutable_data(), frame_data_.data(), frame_data_.size());
        return std::move(arr);
    }

    py::bytes get_memory(unsigned id) {
        void* data = retro_get_memory_data_(id);
        size_t size = retro_get_memory_size_(id);
        if (!data || size == 0) return py::bytes();
        return py::bytes(static_cast<const char*>(data), size);
    }

    py::tuple get_system_info() {
        retro_system_info info = {};
        retro_get_system_info_(&info);
        return py::make_tuple(
            std::string(info.library_name ? info.library_name : ""),
            std::string(info.library_version ? info.library_version : ""));
    }

    py::tuple get_av_info() {
        retro_system_av_info av = {};
        retro_get_system_av_info_(&av);
        return py::make_tuple(av.geometry.base_width, av.geometry.base_height,
                              av.geometry.max_width, av.geometry.max_height,
                              av.timing.fps, av.timing.sample_rate);
    }

private:
    static LibretroCore* instance_;
    void* handle_ = nullptr;
    bool game_loaded_ = false;

    void (*retro_init_)() = nullptr;
    void (*retro_deinit_)() = nullptr;
    void (*retro_run_)() = nullptr;
    void (*retro_reset_)() = nullptr;
    bool (*retro_load_game_)(const retro_game_info*) = nullptr;
    void (*retro_unload_game_)() = nullptr;
    void (*retro_get_system_info_)(retro_system_info*) = nullptr;
    void (*retro_get_system_av_info_)(retro_system_av_info*) = nullptr;
    void* (*retro_get_memory_data_)(unsigned) = nullptr;
    size_t (*retro_get_memory_size_)(unsigned) = nullptr;
    void (*retro_set_environment_)(retro_environment_t) = nullptr;
    void (*retro_set_video_refresh_)(retro_video_refresh_t) = nullptr;
    void (*retro_set_audio_sample_)(retro_audio_sample_t) = nullptr;
    void (*retro_set_audio_sample_batch_)(retro_audio_sample_batch_t) = nullptr;
    void (*retro_set_input_poll_)(retro_input_poll_t) = nullptr;
    void (*retro_set_input_state_)(retro_input_state_t) = nullptr;

    std::vector<uint8_t> game_data_;
    std::vector<uint8_t> frame_data_;
    unsigned frame_w_ = 0, frame_h_ = 0;
    retro_pixel_format pixel_format_ = RETRO_PIXEL_FORMAT_0RGB1555;
    std::string system_dir_, save_dir_;

    // Trampolines into the singleton.
    static bool env_cb(unsigned cmd, void* data) {
        return instance_->handle_env(cmd, data);
    }
    static void video_cb(const void* data, unsigned w, unsigned h, size_t pitch) {
        instance_->handle_video(data, w, h, pitch);
    }
    static void audio_sample_cb(int16_t, int16_t) {}
    static size_t audio_batch_cb(const int16_t*, size_t frames) { return frames; }
    static void input_poll_cb() {}
    static int16_t input_state_cb(unsigned, unsigned, unsigned, unsigned) { return 0; }

    static void log_printf(retro_log_level, const char* fmt, ...) {
        char buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        std::fprintf(stderr, "[core] %s", buf);
    }

    bool handle_env(unsigned cmd, void* data) {
        switch (cmd) {
            case RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
                *static_cast<const char**>(data) = system_dir_.c_str();
                return true;
            case RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
                *static_cast<const char**>(data) = save_dir_.c_str();
                return true;
            case RETRO_ENVIRONMENT_SET_PIXEL_FORMAT: {
                auto fmt = *static_cast<const retro_pixel_format*>(data);
                if (fmt == RETRO_PIXEL_FORMAT_XRGB8888 ||
                    fmt == RETRO_PIXEL_FORMAT_RGB565 ||
                    fmt == RETRO_PIXEL_FORMAT_0RGB1555) {
                    pixel_format_ = fmt;
                    return true;
                }
                return false;
            }
            case RETRO_ENVIRONMENT_GET_CAN_DUPE:
                *static_cast<bool*>(data) = true;
                return true;
            case RETRO_ENVIRONMENT_GET_LOG_INTERFACE: {
                auto* log = static_cast<retro_log_callback*>(data);
                log->log = &LibretroCore::log_printf;
                return true;
            }
            case RETRO_ENVIRONMENT_GET_VARIABLE: {
                auto* v = static_cast<retro_variable*>(data);
                v->value = nullptr;  // use core defaults
                return false;
            }
            case RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
                *static_cast<bool*>(data) = false;
                return true;
            case RETRO_ENVIRONMENT_GET_LANGUAGE:
                *static_cast<unsigned*>(data) = RETRO_LANGUAGE_ENGLISH;
                return true;
            case RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
                *static_cast<unsigned*>(data) = 0;  // legacy variables API
                return true;
            case RETRO_ENVIRONMENT_GET_INPUT_BITMASKS:
                return true;
            default:
                return false;
        }
    }

    void handle_video(const void* data, unsigned w, unsigned h, size_t pitch) {
        if (!data) return;  // duped frame -- keep previous
        frame_w_ = w;
        frame_h_ = h;
        frame_data_.resize(static_cast<size_t>(w) * h * 4);

        if (pixel_format_ == RETRO_PIXEL_FORMAT_XRGB8888) {
            // XRGB8888 little-endian bytes are B G R X. Repack to R G B A.
            const uint8_t* src_row = static_cast<const uint8_t*>(data);
            uint8_t* dst = frame_data_.data();
            for (unsigned y = 0; y < h; ++y) {
                const uint8_t* src = src_row + y * pitch;
                for (unsigned x = 0; x < w; ++x) {
                    uint8_t b = src[0], g = src[1], r = src[2];
                    dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = 0xff;
                    src += 4; dst += 4;
                }
            }
        } else if (pixel_format_ == RETRO_PIXEL_FORMAT_RGB565) {
            const uint16_t* src_row = static_cast<const uint16_t*>(data);
            size_t row_pix = pitch / 2;
            uint8_t* dst = frame_data_.data();
            for (unsigned y = 0; y < h; ++y) {
                const uint16_t* src = src_row + y * row_pix;
                for (unsigned x = 0; x < w; ++x) {
                    uint16_t v = src[x];
                    uint8_t r = (v >> 11) & 0x1f;
                    uint8_t g = (v >> 5)  & 0x3f;
                    uint8_t b =  v        & 0x1f;
                    dst[0] = (r << 3) | (r >> 2);
                    dst[1] = (g << 2) | (g >> 4);
                    dst[2] = (b << 3) | (b >> 2);
                    dst[3] = 0xff;
                    dst += 4;
                }
            }
        } else {  // 0RGB1555
            const uint16_t* src_row = static_cast<const uint16_t*>(data);
            size_t row_pix = pitch / 2;
            uint8_t* dst = frame_data_.data();
            for (unsigned y = 0; y < h; ++y) {
                const uint16_t* src = src_row + y * row_pix;
                for (unsigned x = 0; x < w; ++x) {
                    uint16_t v = src[x];
                    uint8_t r = (v >> 10) & 0x1f;
                    uint8_t g = (v >> 5)  & 0x1f;
                    uint8_t b =  v        & 0x1f;
                    dst[0] = (r << 3) | (r >> 2);
                    dst[1] = (g << 3) | (g >> 2);
                    dst[2] = (b << 3) | (b >> 2);
                    dst[3] = 0xff;
                    dst += 4;
                }
            }
        }
    }
};

LibretroCore* LibretroCore::instance_ = nullptr;

PYBIND11_MODULE(_libretro, m) {
    m.doc() = "Thin libretro frontend exposing one core to Python";

    py::class_<LibretroCore>(m, "LibretroCore")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("core_path"), py::arg("system_dir"), py::arg("save_dir"))
        .def("load_game", &LibretroCore::load_game, py::arg("path"))
        .def("run", &LibretroCore::run)
        .def("reset", &LibretroCore::reset)
        .def("get_frame", &LibretroCore::get_frame)
        .def("get_memory", &LibretroCore::get_memory, py::arg("id") = RETRO_MEMORY_SYSTEM_RAM)
        .def("get_system_info", &LibretroCore::get_system_info)
        .def("get_av_info", &LibretroCore::get_av_info);

    m.attr("MEMORY_SYSTEM_RAM") = py::int_(RETRO_MEMORY_SYSTEM_RAM);
    m.attr("MEMORY_SAVE_RAM")   = py::int_(RETRO_MEMORY_SAVE_RAM);
    m.attr("MEMORY_RTC")        = py::int_(RETRO_MEMORY_RTC);
    m.attr("MEMORY_VIDEO_RAM")  = py::int_(RETRO_MEMORY_VIDEO_RAM);
}
