#include "vema3d_tracker.hpp"
#include "nvdstracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

// C++17 SFINAE trait detectors keep NvMOT bindings portable across DeepStream SDK revisions
// (e.g. NvMOTQuery::batchMode vs batchProcessing, and NvMOTFrame::timeMeta vs timeStamp).
template <typename T, typename = void>
struct has_batch_mode : std::false_type {};
template <typename T>
struct has_batch_mode<T, std::void_t<decltype(std::declval<T>().batchMode)>> : std::true_type {};

template <typename T, typename = void>
struct has_batch_processing : std::false_type {};
template <typename T>
struct has_batch_processing<T, std::void_t<decltype(std::declval<T>().batchProcessing)>> : std::true_type {};

template <typename T, typename = void>
struct has_support_batch_processing : std::false_type {};
template <typename T>
struct has_support_batch_processing<T, std::void_t<decltype(std::declval<T>().supportBatchProcessing)>> : std::true_type {};

template <typename T, typename = void>
struct has_output_shadow_tracks : std::false_type {};
template <typename T>
struct has_output_shadow_tracks<T, std::void_t<decltype(std::declval<T>().outputShadowTracks)>> : std::true_type {};

template <typename T, typename = void>
struct has_max_targets : std::false_type {};
template <typename T>
struct has_max_targets<T, std::void_t<decltype(std::declval<T>().maxTargetsPerStream)>> : std::true_type {};

template <typename T, typename = void>
struct has_max_shadow_age : std::false_type {};
template <typename T>
struct has_max_shadow_age<T, std::void_t<decltype(std::declval<T>().maxShadowTrackingAge)>> : std::true_type {};

template <typename T, typename = void>
struct has_time_meta : std::false_type {};
template <typename T>
struct has_time_meta<T, std::void_t<decltype(std::declval<T>().timeMeta)>> : std::true_type {};

template <typename T, typename = void>
struct has_time_stamp_valid : std::false_type {};
template <typename T>
struct has_time_stamp_valid<T, std::void_t<decltype(std::declval<T>().timeStampValid), decltype(std::declval<T>().timeStamp)>> : std::true_type {};

template <typename T, typename = void>
struct has_src_dims : std::false_type {};
template <typename T>
struct has_src_dims<T, std::void_t<decltype(std::declval<T>().srcFrameWidth), decltype(std::declval<T>().srcFrameHeight)>> : std::true_type {};

template <typename Q>
void set_query_batch_mode(Q* q) {
    if constexpr (has_batch_mode<Q>::value) {
        // NvMOTBatchMode_Batch == 1 across DeepStream SDKs; cast avoids name lookup on legacy headers.
        q->batchMode = static_cast<std::decay_t<decltype(q->batchMode)>>(1);
    } else if constexpr (has_batch_processing<Q>::value) {
        q->batchProcessing = true;
    } else if constexpr (has_support_batch_processing<Q>::value) {
        q->supportBatchProcessing = true;
    }
}

template <typename Q>
void set_query_caps(Q* q, const vema3d::Config& cfg) {
    if constexpr (has_output_shadow_tracks<Q>::value) {
        q->outputShadowTracks = cfg.emit_shadow_tracks;
    }
    if constexpr (has_max_targets<Q>::value) {
        q->maxTargetsPerStream = 256;
    }
    if constexpr (has_max_shadow_age<Q>::value) {
        q->maxShadowTrackingAge = static_cast<uint32_t>(std::max(1, cfg.max_lost_frames));
    }
}

template <typename F>
bool extract_frame_pts_ns(const F& frame, uint64_t& out_ns) {
    if constexpr (has_time_meta<F>::value) {
        if (frame.timeMeta.frameTimeValid) {
            out_ns = static_cast<uint64_t>(frame.timeMeta.frameTimeStamp);
            return true;
        }
        return false;
    } else if constexpr (has_time_stamp_valid<F>::value) {
        if (frame.timeStampValid) {
            out_ns = static_cast<uint64_t>(frame.timeStamp);
            return true;
        }
        return false;
    } else {
        return false;
    }
}

template <typename F>
void extract_frame_dims(const F& frame, int& out_w, int& out_h) {
    if constexpr (has_src_dims<F>::value) {
        if (frame.srcFrameWidth > 0 && frame.srcFrameHeight > 0) {
            out_w = static_cast<int>(frame.srcFrameWidth);
            out_h = static_cast<int>(frame.srcFrameHeight);
            return;
        }
    }
    if (frame.numBuffers > 0 && frame.bufferList != nullptr && frame.bufferList[0] != nullptr) {
        if (frame.bufferList[0]->width > 0 && frame.bufferList[0]->height > 0) {
            out_w = static_cast<int>(frame.bufferList[0]->width);
            out_h = static_cast<int>(frame.bufferList[0]->height);
        }
    }
}

struct StreamContext {
    std::unique_ptr<vema3d::VEMA3DTracker> tracker;
    bool has_last_ts = false;
    uint64_t last_ts_ns = 0;
};

struct NvMotContext {
    vema3d::Config cfg;
    std::mutex mu;
    std::unordered_map<NvMOTStreamId, StreamContext> streams;
};

}  // namespace

extern "C" {

NvMOTStatus NvMOT_Query(
    uint16_t customConfigFilePathSize,
    char* pCustomConfigFilePath,
    NvMOTQuery* pQuery) {
    if (pQuery == nullptr) {
        return NvMOTStatus_Error;
    }
    std::memset(pQuery, 0, sizeof(*pQuery));

    vema3d::Config cfg;
    if (customConfigFilePathSize > 0 && pCustomConfigFilePath != nullptr && pCustomConfigFilePath[0] != '\0') {
        cfg = vema3d::Config::from_yaml_file(std::string(pCustomConfigFilePath));
    }

    pQuery->computeConfig = NVMOTCOMP_CPU;
    pQuery->numTransforms = 0;
    pQuery->supportPastFrame = false;
    set_query_batch_mode(pQuery);
    set_query_caps(pQuery, cfg);
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_Init(
    NvMOTConfig* pConfigIn,
    NvMOTContextHandle* pContextHandle,
    NvMOTConfigResponse* pConfigResponse) {
    if (pConfigIn == nullptr || pContextHandle == nullptr || pConfigResponse == nullptr) {
        return NvMOTStatus_Error;
    }

    auto* ctx = new NvMotContext();
    if (pConfigIn->customConfigFilePathSize > 0 && pConfigIn->customConfigFilePath != nullptr &&
        pConfigIn->customConfigFilePath[0] != '\0') {
        ctx->cfg = vema3d::Config::from_yaml_file(std::string(pConfigIn->customConfigFilePath));
    }

    if (pConfigIn->numTransforms > 0 && pConfigIn->perTransformBatchConfig != nullptr) {
        const auto& tc = pConfigIn->perTransformBatchConfig[0];
        if (tc.maxWidth > 0 && tc.maxHeight > 0) {
            ctx->cfg.img_width = static_cast<int>(tc.maxWidth);
            ctx->cfg.img_height = static_cast<int>(tc.maxHeight);
        }
    }

    pConfigResponse->summaryStatus = NvMOTConfigStatus_OK;
    pConfigResponse->computeStatus = NvMOTConfigStatus_OK;
    pConfigResponse->transformBatchStatus = NvMOTConfigStatus_OK;
    pConfigResponse->miscConfigStatus = NvMOTConfigStatus_OK;
    pConfigResponse->customConfigStatus = NvMOTConfigStatus_OK;

    *pContextHandle = static_cast<NvMOTContextHandle>(ctx);
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_Process(
    NvMOTContextHandle contextHandle,
    NvMOTProcessParams* pParams,
    NvMOTTrackedObjBatch* pTrackedObjectsBatch) {
    if (contextHandle == nullptr || pParams == nullptr || pTrackedObjectsBatch == nullptr) {
        return NvMOTStatus_Error;
    }

    auto* ctx = static_cast<NvMotContext*>(contextHandle);
    std::lock_guard<std::mutex> lock(ctx->mu);

    const uint32_t num_frames = pParams->numFrames;
    pTrackedObjectsBatch->numFilled = num_frames;

    for (uint32_t f_idx = 0; f_idx < num_frames; ++f_idx) {
        const NvMOTFrame& frame = pParams->frameList[f_idx];
        NvMOTTrackedObjList& out_list = pTrackedObjectsBatch->list[f_idx];

        out_list.streamID = frame.streamID;
        out_list.frameNum = frame.frameNum;

        if (!frame.doTracking) {
            out_list.valid = false;
            out_list.numFilled = 0;
            continue;
        }

        auto& sc = ctx->streams[frame.streamID];
        if (!sc.tracker) {
            sc.tracker = std::make_unique<vema3d::VEMA3DTracker>(ctx->cfg);
        }

        if (frame.reset) {
            sc.tracker->reset();
            sc.has_last_ts = false;
            sc.last_ts_ns = 0;
        }

        int frame_w = 0, frame_h = 0;
        extract_frame_dims(frame, frame_w, frame_h);
        if (frame_w > 0 && frame_h > 0) {
            sc.tracker->set_image_size(frame_w, frame_h);
        }

        double dt_sec = -1.0;
        uint64_t ts_ns = 0;
        if (extract_frame_pts_ns(frame, ts_ns)) {
            if (sc.has_last_ts && ts_ns > sc.last_ts_ns) {
                dt_sec = static_cast<double>(ts_ns - sc.last_ts_ns) * 1e-9;
            }
            sc.last_ts_ns = ts_ns;
            sc.has_last_ts = true;
        }

        const uint32_t n_in = frame.objectsIn.numFilled;
        std::vector<vema3d::BBox> dets;
        std::vector<double> scores;
        std::vector<int64_t> cls_ids;
        std::vector<uint32_t> orig_indices;
        dets.reserve(n_in);
        scores.reserve(n_in);
        cls_ids.reserve(n_in);
        orig_indices.reserve(n_in);

        for (uint32_t i = 0; i < n_in; ++i) {
            const NvMOTObjToTrack& obj = frame.objectsIn.list[i];
            if (!obj.doTracking) {
                continue;
            }
            dets.push_back(vema3d::BBox{
                static_cast<double>(obj.bbox.x),
                static_cast<double>(obj.bbox.y),
                static_cast<double>(obj.bbox.x + obj.bbox.width),
                static_cast<double>(obj.bbox.y + obj.bbox.height),
            });
            scores.push_back(static_cast<double>(obj.confidence));
            cls_ids.push_back(static_cast<int64_t>(obj.classId));
            orig_indices.push_back(i);
        }

        auto tracked = sc.tracker->update(dets, scores, cls_ids, dt_sec);
        const uint32_t cap = out_list.numAllocated;
        const uint32_t n_emit = std::min(static_cast<uint32_t>(tracked.size()), cap);
        out_list.valid = true;
        out_list.numFilled = n_emit;

        for (uint32_t i = 0; i < n_emit; ++i) {
            const auto& t = tracked[i];
            NvMOTTrackedObj& dst = out_list.list[i];
            std::memset(&dst, 0, sizeof(dst));

            dst.classId = static_cast<uint16_t>(std::max<int64_t>(0, t.cls_id));
            dst.trackingId = t.track_id;
            dst.age = t.age;

            double bx = t.bbox.x1;
            double by = t.bbox.y1;
            double bw = std::max(0.0, t.bbox.x2 - t.bbox.x1);
            double bh = std::max(0.0, t.bbox.y2 - t.bbox.y1);
            dst.bbox.x = static_cast<float>(bx);
            dst.bbox.y = static_cast<float>(by);
            dst.bbox.width = static_cast<float>(bw);
            dst.bbox.height = static_cast<float>(bh);

            if (t.is_shadow || t.associated_det_idx < 0) {
                // Shadow/coasting contract: nullptr signals nvtracker to emit conf=-0.1 + tracker_confidence.
                dst.associatedObjectIn = nullptr;
                double decay = std::pow(0.95, std::max(0.0, static_cast<double>(t.age) * 0.05));
                dst.confidence = static_cast<float>(std::clamp(t.conf * decay, 0.05, 1.0));
            } else {
                uint32_t raw_idx = orig_indices[static_cast<size_t>(t.associated_det_idx)];
                dst.associatedObjectIn = &frame.objectsIn.list[raw_idx];
                dst.confidence = static_cast<float>(t.conf);
            }
        }
    }

    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_RemoveStreams(
    NvMOTContextHandle contextHandle,
    NvMOTStreamId streamIdMask) {
    if (contextHandle == nullptr) {
        return NvMOTStatus_Error;
    }
    auto* ctx = static_cast<NvMotContext*>(contextHandle);
    std::lock_guard<std::mutex> lock(ctx->mu);
    ctx->streams.erase(streamIdMask);
    return NvMOTStatus_OK;
}

void NvMOT_DeInit(NvMOTContextHandle contextHandle) {
    delete static_cast<NvMotContext*>(contextHandle);
}

}  // extern "C"
