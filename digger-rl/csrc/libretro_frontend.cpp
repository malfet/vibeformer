// Minimal libretro frontend exposed to Python via pybind11.
// Wraps a single libretro core (loaded via dlopen) as a Python class with
// run / load_game / get_frame / get_memory hooks. Singleton: only one core
// can be active per process because libretro callbacks have no userdata.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <array>
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
        RESOLVE(retro_serialize_size);
        RESOLVE(retro_serialize);
        RESOLVE(retro_unserialize);
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
        // The libretro spec says the frontend must call retro_get_system_av_info
        // after retro_load_game so the core knows the frontend has read the
        // timing/geometry. DOSBox Pure relies on this -- skipping it leaves
        // internal state half-initialized and the first retro_run() segfaults.
        retro_system_av_info av = {};
        retro_get_system_av_info_(&av);
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

    void set_key(unsigned retro_key, bool pressed) {
        if (retro_key >= keyboard_state_.size()) return;
        keyboard_state_[retro_key] = pressed;
        // DOSBox Pure registers an event-style keyboard callback (push model)
        // AND polls input_state (pull model). Drive both so either path works.
        if (keyboard_event_cb_) {
            keyboard_event_cb_(pressed, retro_key, 0, 0);
        }
    }

    void clear_keys() {
        for (size_t i = 0; i < keyboard_state_.size(); ++i) {
            if (keyboard_state_[i] && keyboard_event_cb_) {
                keyboard_event_cb_(false, static_cast<unsigned>(i), 0, 0);
            }
            keyboard_state_[i] = false;
        }
    }

    py::list get_held_keys() {
        py::list out;
        for (size_t i = 0; i < keyboard_state_.size(); ++i) {
            if (keyboard_state_[i]) out.append(static_cast<unsigned>(i));
        }
        return out;
    }

    void set_held_keys_raw(const py::list& keys) {
        // Replace the polled-input mirror without firing keyboard events.
        // Intended for use right after unserialize(): the core's *internal*
        // BIOS/DOS keyboard state is whatever was in the saved blob; this
        // function only repairs the mirror used by input_state_cb so polled
        // reads agree with the restored internal state.
        for (size_t i = 0; i < keyboard_state_.size(); ++i) {
            keyboard_state_[i] = false;
        }
        for (auto h : keys) {
            unsigned k = h.cast<unsigned>();
            if (k < keyboard_state_.size()) keyboard_state_[k] = true;
        }
    }

    py::bytes get_memory(unsigned id) {
        void* data = retro_get_memory_data_(id);
        size_t size = retro_get_memory_size_(id);
        if (!data || size == 0) return py::bytes();
        return py::bytes(static_cast<const char*>(data), size);
    }

    size_t serialize_size() {
        return retro_serialize_size_();
    }

    py::bytes serialize() {
        size_t sz = retro_serialize_size_();
        if (sz == 0) {
            throw std::runtime_error(
                "core reports serialize_size==0; state save not supported");
        }
        std::vector<char> buf(sz);
        if (!retro_serialize_(buf.data(), sz)) {
            throw std::runtime_error("retro_serialize() returned false");
        }
        return py::bytes(buf.data(), sz);
    }

    void unserialize(const py::bytes& data) {
        char* buffer;
        py::ssize_t length;
        if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &length) != 0) {
            throw std::runtime_error("failed to read state bytes");
        }
        // libretro's serialize_size() is *not* monotonic-stable: DOSBox
        // Pure grows the reported state size as more emulator structures
        // get allocated (e.g. after extra memory regions are touched).
        // A blob captured earlier is still legitimately replayable as
        // long as the core accepts it -- the spec promises forward
        // compatibility within a session. So we just pass the original
        // byte length through and let retro_unserialize() decide.
        if (!retro_unserialize_(buffer, static_cast<size_t>(length))) {
            throw std::runtime_error(
                "retro_unserialize() returned false (state may be stale or "
                "corrupted; current serialize_size=" +
                std::to_string(retro_serialize_size_()) +
                ", blob length=" + std::to_string(length) + ")");
        }
    }

    py::list get_memory_maps() {
        py::list result;
        for (const auto& r : memory_maps_) {
            py::dict d;
            d["flags"]     = r.flags;
            d["start"]     = r.start;
            d["len"]       = r.len;
            d["addrspace"] = r.addrspace;
            result.append(d);
        }
        return result;
    }

    py::bytes read_memory_region(size_t idx) {
        if (idx >= memory_maps_.size()) {
            throw std::runtime_error("memory region index out of range");
        }
        const auto& r = memory_maps_[idx];
        return py::bytes(static_cast<const char*>(r.ptr), r.len);
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
    size_t (*retro_serialize_size_)() = nullptr;
    bool (*retro_serialize_)(void*, size_t) = nullptr;
    bool (*retro_unserialize_)(const void*, size_t) = nullptr;
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

    // Input state. keyboard_state_ is indexed by RETROK_* values (the enum
    // tops out near 324; 512 is a safe ceiling).
    std::array<bool, 512> keyboard_state_{};
    retro_keyboard_event_t keyboard_event_cb_ = nullptr;

    // Memory maps registered by the core via SET_MEMORY_MAPS. The ptr fields
    // point into the core's own allocations, which stay live for the lifetime
    // of the core. DOSBox Pure registers 2-3 descriptors covering DOS GAME
    // memory, OS memory, and expanded (EMS/XMS) memory.
    struct MemRegion {
        uint64_t flags;
        void* ptr;
        size_t start;
        size_t len;
        std::string addrspace;
    };
    std::vector<MemRegion> memory_maps_;

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
    static int16_t input_state_cb(unsigned port, unsigned device,
                                  unsigned /*index*/, unsigned id) {
        if (port != 0) return 0;
        if (device == RETRO_DEVICE_KEYBOARD) {
            return (id < instance_->keyboard_state_.size() &&
                    instance_->keyboard_state_[id]) ? 1 : 0;
        }
        return 0;
    }

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
            case RETRO_ENVIRONMENT_SET_KEYBOARD_CALLBACK: {
                auto* kc = static_cast<const retro_keyboard_callback*>(data);
                keyboard_event_cb_ = kc ? kc->callback : nullptr;
                return true;
            }
            case RETRO_ENVIRONMENT_SET_MEMORY_MAPS: {
                auto* mm = static_cast<const retro_memory_map*>(data);
                memory_maps_.clear();
                if (mm && mm->descriptors) {
                    for (unsigned i = 0; i < mm->num_descriptors; ++i) {
                        const auto& d = mm->descriptors[i];
                        memory_maps_.push_back({
                            d.flags, d.ptr, d.start, d.len,
                            d.addrspace ? std::string(d.addrspace) : std::string()
                        });
                    }
                }
                return true;
            }
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
        .def("serialize_size", &LibretroCore::serialize_size,
             "Bytes needed to snapshot the current core state.")
        .def("serialize", &LibretroCore::serialize,
             "Return a bytes snapshot of the core's current state. "
             "Pass back to unserialize() to restore.")
        .def("unserialize", &LibretroCore::unserialize, py::arg("data"),
             "Restore a state previously produced by serialize(). The "
             "byte length must match the core's current serialize_size().")
        .def("get_memory_maps", &LibretroCore::get_memory_maps,
             "Return the descriptor list set by the core via SET_MEMORY_MAPS.")
        .def("read_memory_region", &LibretroCore::read_memory_region, py::arg("index"),
             "Return the bytes of the registered memory region at the given index.")
        .def("get_system_info", &LibretroCore::get_system_info)
        .def("get_av_info", &LibretroCore::get_av_info)
        .def("get_held_keys", &LibretroCore::get_held_keys,
             "List of retro_keys currently registered as held in the "
             "frontend mirror (used by input_state_cb).")
        .def("set_held_keys_raw", &LibretroCore::set_held_keys_raw, py::arg("keys"),
             "Set the frontend's held-key mirror to exactly this list, "
             "without firing keyboard_event_cb. Use after unserialize().")
        .def("set_key", &LibretroCore::set_key,
             py::arg("retro_key"), py::arg("pressed"))
        .def("clear_keys", &LibretroCore::clear_keys);

    m.attr("MEMORY_SYSTEM_RAM") = py::int_(RETRO_MEMORY_SYSTEM_RAM);
    m.attr("MEMORY_SAVE_RAM")   = py::int_(RETRO_MEMORY_SAVE_RAM);
    m.attr("MEMORY_RTC")        = py::int_(RETRO_MEMORY_RTC);
    m.attr("MEMORY_VIDEO_RAM")  = py::int_(RETRO_MEMORY_VIDEO_RAM);

    // Subset of retro_key values, exposed as integers under _libretro.RETROK.
    // Any unsigned int in [0, 512) is a valid argument to set_key, so this is
    // just a convenience: extend the list if you need more keys.
    py::module keys = m.def_submodule("RETROK", "libretro key codes (subset)");
#define K(pyname, enumname) \
    keys.attr(pyname) = py::int_(static_cast<unsigned>(RETROK_##enumname))
    K("RETURN", RETURN);    K("SPACE", SPACE);
    K("ESCAPE", ESCAPE);    K("TAB", TAB);
    K("BACKSPACE", BACKSPACE); K("DELETE", DELETE);
    K("LEFT", LEFT);  K("RIGHT", RIGHT);
    K("UP", UP);      K("DOWN", DOWN);
    K("LSHIFT", LSHIFT); K("RSHIFT", RSHIFT);
    K("LCTRL", LCTRL);   K("RCTRL", RCTRL);
    K("LALT", LALT);     K("RALT", RALT);
    K("F1",  F1);  K("F2",  F2);  K("F3",  F3);  K("F4",  F4);
    K("F5",  F5);  K("F6",  F6);  K("F7",  F7);  K("F8",  F8);
    K("F9",  F9);  K("F10", F10); K("F11", F11); K("F12", F12);
    K("A", a); K("B", b); K("C", c); K("D", d); K("E", e);
    K("F", f); K("G", g); K("H", h); K("I", i); K("J", j);
    K("K", k); K("L", l); K("M", m); K("N", n); K("O", o);
    K("P", p); K("Q", q); K("R", r); K("S", s); K("T", t);
    K("U", u); K("V", v); K("W", w); K("X", x); K("Y", y);
    K("Z", z);
    K("N0", 0); K("N1", 1); K("N2", 2); K("N3", 3); K("N4", 4);
    K("N5", 5); K("N6", 6); K("N7", 7); K("N8", 8); K("N9", 9);
#undef K
}
