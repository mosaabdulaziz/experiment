#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vema2 {

struct BBox {
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct TrackOutput {
    uint64_t track_id = 0;
    BBox bbox{};
    double conf = 0.0;
    int64_t cls_id = 0;
    bool is_shadow = false;
    uint32_t age = 0;
    double k_lost = 0.0;
    int associated_det_idx = -1;
};

struct Config {
    // Shared detector & lifecycle contract
    double tau_high = 0.60;
    double tau_low = 0.15;
    double iou_thresh_high = 0.25;
    double iou_thresh_low = 0.15;
    int probation_frames = 2;
    int max_lost_frames = 30;
    int early_term_lost = 2;
    double min_iou_diff_new = 0.70;
    bool check_class_match = true;
    bool emit_shadow_tracks = false;
    double ref_fps = 30.0;
    bool dt_aware = true;

    // Single intrinsic physical parameter (seconds)
    double tau_mem = 0.15;

    // Optional track-creation confidence floor (0 = fall back to tau_high)
    double new_track_thresh = 0.0;

    // DeepStream production lifecycle controls
    int border_exit_lost = 0;
    double border_exit_margin_px = 16.0;
    double min_visible_frac = 0.25;
    int max_shadow_emit_lost = 0;
    int max_targets_per_stream = 120;
    int img_width = 1920;
    int img_height = 1080;

    double effective_new_track_thresh() const {
        return (new_track_thresh > tau_high) ? new_track_thresh : tau_high;
    }

    void validate();
    static Config from_yaml_string(const std::string& content);
    static Config from_yaml_file(const std::string& path);
};

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_rows;
    std::vector<int> unmatched_cols;
};

AssignmentResult linear_assignment(
    const std::vector<double>& cost_matrix,
    int n_rows,
    int n_cols,
    double cost_limit);

enum class TrackState {
    Tentative = 0,
    Confirmed = 1,
};

class PolarEMATrack {
public:
    uint64_t id = 0;
    int64_t cls_id = 0;
    double conf = 0.0;

    double pos_x = 0.0;
    double pos_y = 0.0;
    double d_w = 1.0;
    double d_h = 1.0;

    // Decoupled polar kinematics (speed in px/sec, mean resultant heading vector in unit disk)
    double s_ema = 0.0;
    double m_x = 0.0;
    double m_y = 0.0;

    // Trust-weighted clean anchor (exact constant-velocity invariant)
    double p_anchor_x = 0.0;
    double p_anchor_y = 0.0;
    double t_anchor = 0.0;

    // Continuous visibility/health score H_t in [0, 1] and reference-frame lost counter
    double health = 0.0;
    double k_lost = 0.0;

    TrackState state = TrackState::Tentative;
    int hits = 1;
    uint32_t age = 1;
    bool has_vel = false;

    // Online innovation variance, speed-spread EMA, and scalar NSA position variance
    double inn_sq = 0.0;
    double s_dev = 0.0;
    double p_var = 0.0;

    PolarEMATrack(
        uint64_t track_id,
        const BBox& bbox,
        double conf_in,
        int64_t cls_in,
        double t_sec,
        const Config& cfg);

    double R() const;
    std::pair<double, double> u_hat() const;
    std::pair<double, double> v_ema_sec() const;
    double sigma_noise_px() const;
    double sigma_noise_sec(double dt_sec) const;
    double rho(double dt_sec) const;

    BBox predict(double dt_sec, const Config& cfg);
    void mark_missed(double dt_sec, const Config& cfg);
    void update(
        const BBox& obs_box,
        double obs_conf,
        int64_t obs_cls,
        double t_sec,
        double dt_sec,
        double max_other_iou,
        const Config& cfg);

    BBox get_bbox() const;
};

class PolarEMATracker {
public:
    explicit PolarEMATracker(const Config& cfg = Config{});

    void reset(double fps = -1.0);
    void set_image_size(int width, int height);

    std::vector<TrackOutput> update(
        const std::vector<BBox>& dets,
        const std::vector<double>& scores,
        const std::vector<int64_t>& cls_ids,
        double dt_sec = -1.0);

    const std::vector<PolarEMATrack>& tracks() const { return tracks_; }
    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    std::vector<PolarEMATrack> tracks_;
    uint64_t next_id_ = 1;
    uint64_t frame_idx_ = 0;
    double t_sec_ = 0.0;
};

}  // namespace vema2
