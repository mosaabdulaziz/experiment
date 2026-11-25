#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vema3d {

struct BBox {
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct Config {
    double tau_high = 0.60;
    double tau_low = 0.15;

    double alpha = 0.80;
    double gamma = 0.98;
    double eps_cal = 0.05;
    double v_var_alpha = 0.85;
    double scale_alpha = 0.90;
    double scale_gamma = 0.0;
    double scale_max_rate = 0.05;
    double ref_fps = 30.0;
    bool dt_aware = true;

    double area_ratio_min = 0.75;
    double cos_min = 0.0;
    double min_speed_ratio = 0.05;

    double iou_thresh_high = 0.25;
    double iou_thresh_low = 0.15;
    double relocation_cost_thresh = 1.60;
    double biou_rate = 0.05;
    double biou_max_expansion = 2.0;
    double lambda_biou = 1.0;
    double lambda_pos = 0.35;
    double lambda_dir = 0.25;
    double sigma_par_scale = 1.5;
    double sigma_perp_scale = 0.6;
    double base_pos_sigma_ratio = 0.25;

    int probation_frames = 2;
    int max_lost_frames = 30;
    int early_term_lost = 2;
    double min_iou_diff_new = 0.70;
    bool check_class_match = true;
    bool emit_shadow_tracks = false;
    int border_exit_lost = 0;
    double border_exit_margin_px = 16.0;
    double min_visible_frac = 0.25;
    int max_shadow_emit_lost = 0;

    bool enable_truncation_gate = true;
    bool enable_size_lock = true;
    bool enable_velocity_blending = true;
    bool enable_anisotropic_cost = true;
    bool enable_biou = true;
    bool enable_dir_cost = true;
    bool enable_scale_velocity = true;

    double cam_height = 2.0;
    double cam_pitch_deg = 8.0;
    double min_ray_y = 0.02;
    double max_depth = 80.0;
    double pos_3d_process_std = 0.22;
    double lambda_mmd = 0.45;
    double mmd_gate = 12.0;
    bool auto_pitch = true;

    int img_width = 1920;
    int img_height = 1080;

    static Config from_yaml_string(const std::string& content);
    static Config from_yaml_file(const std::string& path);
};

class PinholeGroundCamera {
public:
    double width = 1920.0;
    double height = 1080.0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double cam_height = 2.0;
    double pitch_deg = 8.0;
    double pitch_rad = 0.0;
    double sin_pitch = 0.0;
    double cos_pitch = 1.0;
    double min_ray_y = 0.02;
    double max_depth = 80.0;

    PinholeGroundCamera(
        double width = 1920.0,
        double height = 1080.0,
        double cam_height = 2.0,
        double pitch_deg = 8.0,
        double min_ray_y = 0.02,
        double max_depth = 80.0,
        double fx_opt = -1.0,
        double fy_opt = -1.0,
        double cx_opt = -1.0,
        double cy_opt = -1.0);

    std::array<double, 4> jacobian_2d_to_3d(double u, double v, double dh) const;
    std::array<double, 4> default_cov_2d(double w_px, double h_px, double conf_scale = 1.0) const;

    void intersect_plane(
        double u,
        double v,
        double dh,
        const std::array<double, 4>& cov_2d,
        bool is_head,
        bool has_fallback_z,
        double fallback_z,
        std::array<double, 2>& out_pos,
        std::array<double, 4>& out_cov,
        bool& out_clamped) const;

    void head_bbox_to_pole(
        const BBox& box,
        double pole_h,
        const std::array<double, 4>& cov_2d,
        double fallback_z,
        double base_elev,
        std::array<double, 2>& out_pos,
        std::array<double, 4>& out_cov) const;

    void bbox_to_pole_elev(
        const BBox& box,
        const std::array<double, 4>& cov_2d,
        std::array<double, 2>& out_pos,
        double& out_pole_h,
        double& out_pole_w,
        std::array<double, 4>& out_cov,
        double& out_base_elev) const;

    void bbox_to_pole_batch(
        const std::vector<BBox>& boxes,
        const std::vector<double>& confs,
        std::vector<std::array<double, 2>>& out_pos,
        std::vector<std::array<double, 4>>& out_cov) const;

    BBox project_pole_to_bbox(
        const std::array<double, 2>& pos_3d,
        double pole_h,
        double pole_w,
        double base_elev = 0.0) const;

    PinholeGroundCamera auto_pitch_for_bboxes(
        const std::vector<BBox>& boxes,
        double min_margin_ray_y = 0.05,
        double max_pitch_deg = 35.0) const;

private:
    void pixel_to_world_ray(double u, double v, double& rx, double& ry, double& rz) const;
};

enum class TrackState {
    Tentative = 0,
    Confirmed = 1,
    Lost = 2,
};

struct TrackOutput {
    uint64_t track_id = 0;
    BBox bbox{};
    double conf = 0.0;
    int64_t cls_id = 0;
    bool is_shadow = false;
    uint32_t age = 0;
    int associated_det_idx = -1;
};

class VEMA3DTrack {
public:
    uint64_t id = 0;
    int64_t cls_id = 0;
    double conf = 0.0;
    std::array<double, 2> pos_3d{0.0, 0.0};
    std::array<double, 2> v_3d{0.0, 0.0};
    double pole_h = 0.0;
    double pole_w = 0.0;
    double base_elev = 0.0;
    std::array<double, 4> cov_pos{1.0, 0.0, 0.0, 1.0};
    double v_var_3d = 0.0;

    bool has_vel = false;
    std::array<double, 2> last_obs_3d{0.0, 0.0};
    BBox last_obs_bbox{};
    double last_obs_time = 0.0;
    double time_now = 0.0;

    double k_lost = 0.0;
    int k_degraded = 0;
    bool last_anchor_head = false;
    int hits = 1;
    uint32_t age = 1;
    double last_beta = 1.0;
    TrackState state = TrackState::Tentative;
    int associated_det_idx = -1;

    VEMA3DTrack(
        uint64_t track_id,
        const BBox& bbox_xyxy,
        double conf,
        int64_t cls_id,
        const Config& cfg,
        const PinholeGroundCamera& camera,
        int det_idx = -1);

    BBox predict(double dt_steps);
    void mark_missed(double dt_steps);
    void update(const BBox& obs_box, double obs_conf, int64_t cls_id, int det_idx);
    BBox get_bbox(bool buffered = false) const;

    void set_camera(const PinholeGroundCamera* cam) { camera_ = cam; }

private:
    const Config* cfg_;
    const PinholeGroundCamera* camera_;
    double pre_predict_pole_h_ = 0.0;
    double pre_predict_pole_w_ = 0.0;
    double pre_predict_elev_ = 0.0;
    int consec_high_nis_ = 0;

    std::array<double, 4> obs_cov_2d(const BBox& obs_box, double obs_conf) const;
    std::array<double, 2> kalman_pos_update(
        const std::array<double, 2>& obs_3d,
        const std::array<double, 4>& cov_obs,
        bool allow_nis_boost);
    bool is_leg_occlusion(const BBox& obs_box, const BBox& pred_box) const;
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

class VEMA3DTracker {
public:
    explicit VEMA3DTracker(const Config& cfg = Config{});

    void reset(double fps = -1.0, int img_width = -1, int img_height = -1);
    void set_image_size(int img_width, int img_height);

    std::vector<TrackOutput> update(
        const std::vector<BBox>& dets,
        const std::vector<double>& scores,
        const std::vector<int64_t>& cls_ids,
        double dt_sec = -1.0);

    const Config& config() const { return cfg_; }
    uint64_t frame_idx() const { return frame_idx_; }

private:
    Config cfg_;
    PinholeGroundCamera camera_;
    std::vector<VEMA3DTrack> tracks_;
    uint64_t next_id_ = 1;
    uint64_t frame_idx_ = 0;

    void refresh_track_camera_ptrs();
};

}  // namespace vema3d

extern "C" {
void* vema3d_test_create(const char* yaml_path, int img_width, int img_height);
int vema3d_test_step(
    void* handle,
    uint64_t stream_id,
    const double* dets_xyxy,
    const double* scores,
    const int64_t* cls_ids,
    int num_dets,
    double dt_sec,
    uint64_t* out_ids,
    double* out_boxes_xyxy,
    double* out_confs,
    int64_t* out_cls_ids,
    int* out_is_shadow,
    int max_out);
int vema3d_test_remove_stream(void* handle, uint64_t stream_id);
void vema3d_test_destroy(void* handle);
}
