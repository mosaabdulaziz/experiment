#include "vema3d_tracker.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace vema3d {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline double deg_to_rad(double deg) {
    return deg * (kPi / 180.0);
}

inline double rad_to_deg(double rad) {
    return rad * (180.0 / kPi);
}

inline std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

inline std::string strip_yaml_comment(const std::string& line) {
    bool in_single = false;
    bool in_double = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\'' && !in_double) {
            in_single = !in_single;
        } else if (c == '"' && !in_single) {
            in_double = !in_double;
        } else if (c == '#' && !in_single && !in_double) {
            return line.substr(0, i);
        }
    }
    return line;
}

inline std::string unquote_copy(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

inline bool parse_bool_val(const std::string& raw, bool fallback) {
    std::string low;
    low.reserve(raw.size());
    for (char c : raw) low.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (low == "true" || low == "1" || low == "yes" || low == "on") return true;
    if (low == "false" || low == "0" || low == "no" || low == "off") return false;
    return fallback;
}

inline double parse_double_val(const std::string& raw, double fallback) {
    try {
        size_t idx = 0;
        double v = std::stod(raw, &idx);
        return (idx == raw.size()) ? v : fallback;
    } catch (...) {
        return fallback;
    }
}

inline int parse_int_val(const std::string& raw, int fallback) {
    try {
        size_t idx = 0;
        int v = std::stoi(raw, &idx);
        return (idx == raw.size()) ? v : fallback;
    } catch (...) {
        return fallback;
    }
}

inline void inv_2x2(const std::array<double, 4>& s, double& out_det, std::array<double, 4>& out_inv) {
    double s00 = s[0];
    double s01 = s[1];
    double s11 = s[3];
    out_det = std::max(s00 * s11 - s01 * s01, 1e-12);
    out_inv = {s11 / out_det, -s01 / out_det, -s01 / out_det, s00 / out_det};
}

inline double mahalanobis_sq_2x2(const std::array<double, 2>& diff, const std::array<double, 4>& cov) {
    double s00 = cov[0];
    double s01 = cov[1];
    double s11 = cov[3];
    double det = std::max(s00 * s11 - s01 * s01, 1e-12);
    double dx = diff[0];
    double dz = diff[1];
    return (s11 * dx * dx - 2.0 * s01 * dx * dz + s00 * dz * dz) / det;
}

inline double box_iou(const BBox& a, const BBox& b) {
    double x1 = std::max(a.x1, b.x1);
    double y1 = std::max(a.y1, b.y1);
    double x2 = std::min(a.x2, b.x2);
    double y2 = std::min(a.y2, b.y2);
    double iw = std::max(0.0, x2 - x1);
    double ih = std::max(0.0, y2 - y1);
    double inter = iw * ih;
    double area_a = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
    double area_b = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
    double uni = area_a + area_b - inter;
    return (uni > 1e-9) ? (inter / uni) : 0.0;
}

}  // namespace

Config Config::from_yaml_string(const std::string& content) {
    Config cfg;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        std::string cleaned = trim_copy(strip_yaml_comment(line));
        if (cleaned.empty() || cleaned[0] == '%' || cleaned == "---" || cleaned == "...") {
            continue;
        }
        size_t colon = cleaned.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = trim_copy(cleaned.substr(0, colon));
        std::string val = unquote_copy(trim_copy(cleaned.substr(colon + 1)));
        if (val.empty()) {
            continue;  // section header such as TargetManagement:
        }

        if (key == "tau_high") cfg.tau_high = parse_double_val(val, cfg.tau_high);
        else if (key == "tau_low") cfg.tau_low = parse_double_val(val, cfg.tau_low);
        else if (key == "alpha") cfg.alpha = parse_double_val(val, cfg.alpha);
        else if (key == "gamma") cfg.gamma = parse_double_val(val, cfg.gamma);
        else if (key == "eps_cal") cfg.eps_cal = parse_double_val(val, cfg.eps_cal);
        else if (key == "v_var_alpha") cfg.v_var_alpha = parse_double_val(val, cfg.v_var_alpha);
        else if (key == "scale_alpha") cfg.scale_alpha = parse_double_val(val, cfg.scale_alpha);
        else if (key == "scale_gamma") cfg.scale_gamma = parse_double_val(val, cfg.scale_gamma);
        else if (key == "scale_max_rate") cfg.scale_max_rate = parse_double_val(val, cfg.scale_max_rate);
        else if (key == "ref_fps") cfg.ref_fps = parse_double_val(val, cfg.ref_fps);
        else if (key == "dt_aware") cfg.dt_aware = parse_bool_val(val, cfg.dt_aware);
        else if (key == "area_ratio_min") cfg.area_ratio_min = parse_double_val(val, cfg.area_ratio_min);
        else if (key == "cos_min") cfg.cos_min = parse_double_val(val, cfg.cos_min);
        else if (key == "min_speed_ratio") cfg.min_speed_ratio = parse_double_val(val, cfg.min_speed_ratio);
        else if (key == "iou_thresh_high") cfg.iou_thresh_high = parse_double_val(val, cfg.iou_thresh_high);
        else if (key == "iou_thresh_low") cfg.iou_thresh_low = parse_double_val(val, cfg.iou_thresh_low);
        else if (key == "relocation_cost_thresh") cfg.relocation_cost_thresh = parse_double_val(val, cfg.relocation_cost_thresh);
        else if (key == "biou_rate") cfg.biou_rate = parse_double_val(val, cfg.biou_rate);
        else if (key == "biou_max_expansion") cfg.biou_max_expansion = parse_double_val(val, cfg.biou_max_expansion);
        else if (key == "lambda_biou") cfg.lambda_biou = parse_double_val(val, cfg.lambda_biou);
        else if (key == "lambda_pos") cfg.lambda_pos = parse_double_val(val, cfg.lambda_pos);
        else if (key == "lambda_dir") cfg.lambda_dir = parse_double_val(val, cfg.lambda_dir);
        else if (key == "sigma_par_scale") cfg.sigma_par_scale = parse_double_val(val, cfg.sigma_par_scale);
        else if (key == "sigma_perp_scale") cfg.sigma_perp_scale = parse_double_val(val, cfg.sigma_perp_scale);
        else if (key == "base_pos_sigma_ratio") cfg.base_pos_sigma_ratio = parse_double_val(val, cfg.base_pos_sigma_ratio);
        else if (key == "probation_frames") cfg.probation_frames = parse_int_val(val, cfg.probation_frames);
        else if (key == "max_lost_frames") cfg.max_lost_frames = parse_int_val(val, cfg.max_lost_frames);
        else if (key == "early_term_lost") cfg.early_term_lost = parse_int_val(val, cfg.early_term_lost);
        else if (key == "min_iou_diff_new") cfg.min_iou_diff_new = parse_double_val(val, cfg.min_iou_diff_new);
        else if (key == "check_class_match") cfg.check_class_match = parse_bool_val(val, cfg.check_class_match);
        else if (key == "emit_shadow_tracks") cfg.emit_shadow_tracks = parse_bool_val(val, cfg.emit_shadow_tracks);
        else if (key == "border_exit_lost") cfg.border_exit_lost = parse_int_val(val, cfg.border_exit_lost);
        else if (key == "border_exit_margin_px") cfg.border_exit_margin_px = parse_double_val(val, cfg.border_exit_margin_px);
        else if (key == "min_visible_frac") cfg.min_visible_frac = parse_double_val(val, cfg.min_visible_frac);
        else if (key == "max_shadow_emit_lost") cfg.max_shadow_emit_lost = parse_int_val(val, cfg.max_shadow_emit_lost);
        else if (key == "enable_truncation_gate") cfg.enable_truncation_gate = parse_bool_val(val, cfg.enable_truncation_gate);
        else if (key == "enable_size_lock") cfg.enable_size_lock = parse_bool_val(val, cfg.enable_size_lock);
        else if (key == "enable_velocity_blending") cfg.enable_velocity_blending = parse_bool_val(val, cfg.enable_velocity_blending);
        else if (key == "enable_anisotropic_cost") cfg.enable_anisotropic_cost = parse_bool_val(val, cfg.enable_anisotropic_cost);
        else if (key == "enable_biou") cfg.enable_biou = parse_bool_val(val, cfg.enable_biou);
        else if (key == "enable_dir_cost") cfg.enable_dir_cost = parse_bool_val(val, cfg.enable_dir_cost);
        else if (key == "enable_scale_velocity") cfg.enable_scale_velocity = parse_bool_val(val, cfg.enable_scale_velocity);
        else if (key == "cam_height") cfg.cam_height = parse_double_val(val, cfg.cam_height);
        else if (key == "cam_pitch_deg") cfg.cam_pitch_deg = parse_double_val(val, cfg.cam_pitch_deg);
        else if (key == "min_ray_y") cfg.min_ray_y = parse_double_val(val, cfg.min_ray_y);
        else if (key == "max_depth") cfg.max_depth = parse_double_val(val, cfg.max_depth);
        else if (key == "pos_3d_process_std") cfg.pos_3d_process_std = parse_double_val(val, cfg.pos_3d_process_std);
        else if (key == "lambda_mmd") cfg.lambda_mmd = parse_double_val(val, cfg.lambda_mmd);
        else if (key == "mmd_gate") cfg.mmd_gate = parse_double_val(val, cfg.mmd_gate);
        else if (key == "auto_pitch") cfg.auto_pitch = parse_bool_val(val, cfg.auto_pitch);
        else if (key == "img_width") cfg.img_width = parse_int_val(val, cfg.img_width);
        else if (key == "img_height") cfg.img_height = parse_int_val(val, cfg.img_height);
    }
    cfg.probation_frames = std::max(1, cfg.probation_frames);
    cfg.max_lost_frames = std::max(1, cfg.max_lost_frames);
    cfg.ref_fps = std::max(1e-3, cfg.ref_fps);
    return cfg;
}

Config Config::from_yaml_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return Config{};
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return from_yaml_string(oss.str());
}

PinholeGroundCamera::PinholeGroundCamera(
    double w,
    double h,
    double ch,
    double p_deg,
    double mry,
    double md,
    double fx_opt,
    double fy_opt,
    double cx_opt,
    double cy_opt)
    : width(w),
      height(h),
      cam_height(std::max(ch, 0.1)),
      pitch_deg(p_deg),
      min_ray_y(std::max(mry, 1e-4)),
      max_depth(std::max(md, 1.0)) {
    double f_default = 0.95 * std::max(width, height);
    fx = (fx_opt > 0.0) ? fx_opt : f_default;
    fy = (fy_opt > 0.0) ? fy_opt : f_default;
    cx = (cx_opt >= 0.0) ? cx_opt : 0.5 * width;
    cy = (cy_opt >= 0.0) ? cy_opt : 0.5 * height;
    pitch_rad = deg_to_rad(pitch_deg);
    sin_pitch = std::sin(pitch_rad);
    cos_pitch = std::cos(pitch_rad);
}

void PinholeGroundCamera::pixel_to_world_ray(double u, double v, double& rx, double& ry, double& rz) const {
    double tx = (u - cx) / fx;
    double ty = (v - cy) / fy;
    rx = tx;
    ry = ty * cos_pitch + sin_pitch;
    rz = -ty * sin_pitch + cos_pitch;
}

std::array<double, 4> PinholeGroundCamera::jacobian_2d_to_3d(double u, double v, double dh_val) const {
    double rx = 0.0, ry = 0.0, rz = 0.0;
    pixel_to_world_ray(u, v, rx, ry, rz);
    double rz_safe = std::max(rz, 1e-4);

    if (std::abs(dh_val) <= 0.05 || dh_val * ry <= 0.0) {
        double ry_safe = std::max(std::abs(ry), min_ray_y);
        double z_approx = std::min(std::max(std::abs(dh_val), 0.1) / ry_safe * rz_safe, max_depth);
        return {z_approx / (rz_safe * fx), 0.0, 0.0, 0.0};
    }

    double sign_dh = (dh_val > 0.0) ? 1.0 : -1.0;
    double ry_floor = std::max(std::abs(dh_val) / max_depth, 1e-4);
    double ry_eff = sign_dh * std::max(std::abs(ry), ry_floor);
    double z_c = dh_val / ry_eff;
    double x_val = z_c * rx;

    double j00 = dh_val / (ry_eff * fx);
    double j01 = -(x_val * cos_pitch) / (ry_eff * fy);
    double j11 = -dh_val / ((ry_eff * ry_eff) * fy);
    return {j00, j01, 0.0, j11};
}

std::array<double, 4> PinholeGroundCamera::default_cov_2d(double w_px, double h_px, double conf_scale) const {
    double sigma_u = std::max(2.0, 0.008 * w_px) * conf_scale;
    double sigma_v = std::max(2.0, 0.006 * h_px) * conf_scale;
    return {sigma_u * sigma_u, 0.0, 0.0, sigma_v * sigma_v};
}

void PinholeGroundCamera::intersect_plane(
    double u,
    double v,
    double dh,
    const std::array<double, 4>& cov_2d,
    bool is_head,
    bool has_fallback_z,
    double fallback_z,
    std::array<double, 2>& out_pos,
    std::array<double, 4>& out_cov,
    bool& out_clamped) const {
    double rx = 0.0, ry = 0.0, rz = 0.0;
    pixel_to_world_ray(u, v, rx, ry, rz);
    double rz_safe = std::max(rz, 1e-4);
    out_clamped = false;

    auto mul_j_cov_jt = [&](const std::array<double, 4>& j) {
        // J is upper-triangular (j10 == 0).
        double j00 = j[0], j01 = j[1], j11 = j[3];
        double a00 = j00 * cov_2d[0] + j01 * cov_2d[2];
        double a01 = j00 * cov_2d[1] + j01 * cov_2d[3];
        double a10 = j11 * cov_2d[2];
        double a11 = j11 * cov_2d[3];
        double c00 = a00 * j00 + a01 * j01;
        double c01 = a01 * j11;
        double c10 = a10 * j00 + a11 * j01;
        double c11 = a11 * j11;
        double sym01 = 0.5 * (c01 + c10);
        return std::array<double, 4>{c00, sym01, sym01, c11};
    };

    if (!is_head) {
        double ry_safe = ry;
        if (ry < min_ray_y) {
            out_clamped = true;
            ry_safe = min_ray_y;
        }
        double z_c = dh / ry_safe;
        double z_val = z_c * rz_safe;
        double x_val = 0.0;
        if (z_val > max_depth) {
            out_clamped = true;
            z_val = max_depth;
            x_val = z_val * (rx / rz_safe);
        } else if (z_val < 0.3) {
            out_clamped = true;
            z_val = 0.3;
            x_val = z_val * (rx / rz_safe);
        } else {
            x_val = z_c * rx;
        }

        auto j_mat = jacobian_2d_to_3d(u, v, dh);
        if (ry < min_ray_y) {
            double v_clamped = cy + fy * ((min_ray_y - sin_pitch) / std::max(cos_pitch, 1e-4));
            j_mat = jacobian_2d_to_3d(u, v_clamped, dh);
        }
        out_cov = mul_j_cov_jt(j_mat);
        if (out_clamped) {
            out_cov[3] = std::max(out_cov[3], (0.35 * z_val) * (0.35 * z_val));
        }
        out_cov[0] = std::max(out_cov[0], 1e-6);
        out_cov[3] = std::max(out_cov[3], 1e-6);
        out_pos = {x_val, z_val};
        return;
    }

    constexpr double dh_eps = 0.05;
    if (std::abs(dh) <= dh_eps) {
        out_clamped = true;
        double raw_z = has_fallback_z ? fallback_z : std::min(max_depth, 10.0);
        double z_val = std::clamp(raw_z, 0.3, max_depth);
        double x_val = z_val * (rx / rz_safe);
        double j_lat = z_val / (rz_safe * fx);
        out_pos = {x_val, z_val};
        out_cov = {
            (j_lat * j_lat) * cov_2d[0],
            0.0,
            0.0,
            std::max(25.0, (0.5 * z_val) * (0.5 * z_val)),
        };
        return;
    }

    double ry_limit = std::max(std::abs(dh) / max_depth, 1e-4);
    bool valid_sign = (dh > 0.0 && ry >= ry_limit) || (dh < 0.0 && ry <= -ry_limit);
    double x_val = 0.0;
    double z_val = 0.0;
    if (valid_sign) {
        double z_c = dh / ry;
        z_val = z_c * rz_safe;
        if (z_val > max_depth) {
            out_clamped = true;
            z_val = max_depth;
            x_val = z_val * (rx / rz_safe);
        } else if (z_val < 0.3) {
            out_clamped = true;
            z_val = 0.3;
            x_val = z_val * (rx / rz_safe);
        } else {
            x_val = z_c * rx;
        }
        auto j_mat = jacobian_2d_to_3d(u, v, dh);
        out_cov = mul_j_cov_jt(j_mat);
        if (out_clamped) {
            out_cov[3] = std::max(out_cov[3], (0.35 * z_val) * (0.35 * z_val));
        }
    } else {
        out_clamped = true;
        double raw_z = has_fallback_z ? fallback_z : max_depth;
        z_val = std::clamp(raw_z, 0.3, max_depth);
        x_val = z_val * (rx / rz_safe);
        double j_lat = z_val / (rz_safe * fx);
        out_cov = {
            (j_lat * j_lat) * cov_2d[0],
            0.0,
            0.0,
            std::max(25.0, (0.5 * z_val) * (0.5 * z_val)),
        };
    }

    out_cov[0] = std::max(out_cov[0], 1e-6);
    out_cov[3] = std::max(out_cov[3], 1e-6);
    out_pos = {x_val, z_val};
}

void PinholeGroundCamera::head_bbox_to_pole(
    const BBox& box,
    double pole_h,
    const std::array<double, 4>& cov_2d,
    double fallback_z,
    double base_elev,
    std::array<double, 2>& out_pos,
    std::array<double, 4>& out_cov) const {
    double u_cx = 0.5 * (box.x1 + box.x2);
    double dh = cam_height - pole_h;
    bool clamped = false;
    std::array<double, 2> pos_head{};
    intersect_plane(u_cx, box.y1, dh, cov_2d, true, true, fallback_z, pos_head, out_cov, clamped);
    double z_val = pos_head[1];
    double z_c_foot = std::max((cam_height - base_elev) * sin_pitch + z_val * cos_pitch, 0.2);
    double x_foot_aligned = z_c_foot * ((u_cx - cx) / fx);
    out_pos = {x_foot_aligned, z_val};
}

void PinholeGroundCamera::bbox_to_pole_elev(
    const BBox& box,
    const std::array<double, 4>& cov_2d,
    std::array<double, 2>& out_pos,
    double& out_pole_h,
    double& out_pole_w,
    std::array<double, 4>& out_cov,
    double& out_base_elev) const {
    double w_px = std::max(box.x2 - box.x1, 1e-3);
    double u_cx = 0.5 * (box.x1 + box.x2);
    double v_foot = box.y2;
    double v_head = box.y1;

    bool clamped = false;
    intersect_plane(u_cx, v_foot, cam_height, cov_2d, false, false, 0.0, out_pos, out_cov, clamped);
    double z_val = out_pos[1];

    if (clamped) {
        double rx_f = 0.0, ry_f = 0.0, rz_f = 0.0;
        pixel_to_world_ray(u_cx, v_foot, rx_f, ry_f, rz_f);
        double rz_f_safe = std::max(rz_f, 1e-4);
        out_base_elev = cam_height - z_val * (ry_f / rz_f_safe);
    } else {
        out_base_elev = 0.0;
    }

    double rx_h = 0.0, ry_h = 0.0, rz_h = 0.0;
    pixel_to_world_ray(u_cx, v_head, rx_h, ry_h, rz_h);
    double rz_h_safe = std::max(rz_h, 1e-4);
    double head_elev = cam_height - z_val * (ry_h / rz_h_safe);
    out_pole_h = std::max(head_elev - out_base_elev, 0.05);

    double z_c_mid = std::max((cam_height - out_base_elev - 0.5 * out_pole_h) * sin_pitch + z_val * cos_pitch, 0.2);
    out_pole_w = std::max(w_px * z_c_mid / fx, 0.02);
}

void PinholeGroundCamera::bbox_to_pole_batch(
    const std::vector<BBox>& boxes,
    const std::vector<double>& confs,
    std::vector<std::array<double, 2>>& out_pos,
    std::vector<std::array<double, 4>>& out_cov) const {
    const size_t n = boxes.size();
    out_pos.resize(n);
    out_cov.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const auto& b = boxes[i];
        double w_px = std::max(b.x2 - b.x1, 1e-3);
        double h_px = std::max(b.y2 - b.y1, 1e-3);
        double u_cx = 0.5 * (b.x1 + b.x2);

        double tx = (u_cx - cx) / fx;
        double ty_f = (b.y2 - cy) / fy;
        double rx = tx;
        double ry_f = ty_f * cos_pitch + sin_pitch;
        double rz_f = std::max(-ty_f * sin_pitch + cos_pitch, 1e-4);

        bool clamped_ry = ry_f < min_ray_y;
        double ry_safe = std::max(ry_f, min_ray_y);
        double z_c = cam_height / ry_safe;
        double z_val = z_c * rz_f;

        bool over_max = z_val > max_depth;
        bool under_min = z_val < 0.3;
        bool clamped_any = clamped_ry || over_max || under_min;
        z_val = std::clamp(z_val, 0.3, max_depth);
        double x_val = (over_max || under_min) ? z_val * (rx / rz_f) : z_c * rx;
        out_pos[i] = {x_val, z_val};

        double j00 = cam_height / (ry_safe * fx);
        double j01 = -(x_val * cos_pitch) / (ry_safe * fy);
        double j11 = -cam_height / ((ry_safe * ry_safe) * fy);

        double conf_scale = 1.0 + std::max(0.0, 1.0 - confs[i]) * 0.8;
        double su = std::max(2.0, 0.008 * w_px) * conf_scale;
        double sv = std::max(2.0, 0.006 * h_px) * conf_scale;
        double var_u = su * su;
        double var_v = sv * sv;

        double c00 = (j00 * j00) * var_u + (j01 * j01) * var_v;
        double c01 = j01 * j11 * var_v;
        double c11 = (j11 * j11) * var_v;
        if (clamped_any) {
            c11 = std::max(c11, (0.35 * z_val) * (0.35 * z_val));
        }
        c00 = std::max(c00, 1e-6);
        c11 = std::max(c11, 1e-6);
        out_cov[i] = {c00, c01, c01, c11};
    }
}

BBox PinholeGroundCamera::project_pole_to_bbox(
    const std::array<double, 2>& pos_3d,
    double pole_h,
    double pole_w,
    double base_elev) const {
    double x_val = pos_3d[0];
    double z_val = std::max(pos_3d[1], 0.2);
    double ph = std::max(pole_h, 0.02);
    double pw = std::max(pole_w, 0.01);

    double dh_foot = cam_height - base_elev;
    double z_c_foot = std::max(dh_foot * sin_pitch + z_val * cos_pitch, 0.2);
    double u_cx = fx * (x_val / z_c_foot) + cx;
    double v_foot = fy * ((dh_foot * cos_pitch - z_val * sin_pitch) / z_c_foot) + cy;

    double dh_head = cam_height - base_elev - ph;
    double z_c_head = std::max(dh_head * sin_pitch + z_val * cos_pitch, 0.2);
    double v_head = fy * ((dh_head * cos_pitch - z_val * sin_pitch) / z_c_head) + cy;

    double z_c_mid = std::max((dh_foot - 0.5 * ph) * sin_pitch + z_val * cos_pitch, 0.2);
    double w_px = fx * (pw / z_c_mid);

    return BBox{u_cx - 0.5 * w_px, v_head, u_cx + 0.5 * w_px, v_foot};
}

PinholeGroundCamera PinholeGroundCamera::auto_pitch_for_bboxes(
    const std::vector<BBox>& boxes,
    double min_margin_ray_y,
    double max_pitch_deg) const {
    if (boxes.empty()) {
        return *this;
    }
    double min_y2 = boxes[0].y2;
    for (size_t i = 1; i < boxes.size(); ++i) {
        min_y2 = std::min(min_y2, boxes[i].y2);
    }
    double ty_min = (min_y2 - cy) / fy;
    double ry_min = ty_min * cos_pitch + sin_pitch;
    if (ry_min >= min_ray_y * 1.5) {
        return *this;
    }

    double amp = std::hypot(ty_min, 1.0);
    double ratio = std::clamp(min_margin_ray_y / std::max(amp, 1e-6), -0.999, 0.999);
    double needed_rad = std::asin(ratio) - std::atan2(ty_min, 1.0);
    double needed_deg = std::clamp(rad_to_deg(needed_rad), pitch_deg, max_pitch_deg);
    if (needed_deg <= pitch_deg + 0.5) {
        return *this;
    }
    return PinholeGroundCamera(width, height, cam_height, needed_deg, min_ray_y, max_depth, fx, fy, cx, cy);
}

VEMA3DTrack::VEMA3DTrack(
    uint64_t track_id,
    const BBox& bbox_xyxy,
    double init_conf,
    int64_t init_cls,
    const Config& cfg,
    const PinholeGroundCamera& camera,
    int det_idx)
    : id(track_id),
      cls_id(init_cls),
      conf(init_conf),
      last_obs_bbox(bbox_xyxy),
      state(cfg.probation_frames <= 1 ? TrackState::Confirmed : TrackState::Tentative),
      associated_det_idx(det_idx),
      cfg_(&cfg),
      camera_(&camera) {
    auto c2d = obs_cov_2d(bbox_xyxy, conf);
    camera_->bbox_to_pole_elev(bbox_xyxy, c2d, pos_3d, pole_h, pole_w, cov_pos, base_elev);
    last_obs_3d = pos_3d;
    pre_predict_pole_h_ = pole_h;
    pre_predict_pole_w_ = pole_w;
    pre_predict_elev_ = base_elev;
}

std::array<double, 4> VEMA3DTrack::obs_cov_2d(const BBox& obs_box, double obs_conf) const {
    double w_px = std::max(obs_box.x2 - obs_box.x1, 1e-3);
    double h_px = std::max(obs_box.y2 - obs_box.y1, 1e-3);
    double conf_scale = 1.0 + std::max(0.0, 1.0 - obs_conf) * 0.8;
    double su = std::max(2.0, 0.008 * w_px) * conf_scale;
    double sv = std::max(2.0, 0.006 * h_px) * conf_scale;
    return {su * su, 0.0, 0.0, sv * sv};
}

BBox VEMA3DTrack::predict(double dt_steps) {
    dt_steps = std::max(dt_steps, 1e-4);
    associated_det_idx = -1;
    pre_predict_pole_h_ = pole_h;
    pre_predict_pole_w_ = pole_w;
    pre_predict_elev_ = base_elev;

    if (has_vel) {
        double damping = std::pow(cfg_->gamma, k_lost);
        pos_3d[0] += damping * v_3d[0] * dt_steps;
        pos_3d[1] = std::clamp(pos_3d[1] + damping * v_3d[1] * dt_steps, 0.3, camera_->max_depth);
    }

    double fps_scale = 30.0 / std::max(cfg_->ref_fps, 5.0);
    double q_base = (cfg_->pos_3d_process_std * fps_scale);
    q_base = (q_base * q_base) * dt_steps;
    double q_vel = 0.25 * std::max(v_var_3d, 0.0) * (dt_steps * dt_steps);
    cov_pos[0] += q_base + q_vel;
    cov_pos[3] += q_base + q_vel;

    double sym01 = 0.5 * (cov_pos[1] + cov_pos[2]);
    cov_pos[0] = std::clamp(cov_pos[0], 1e-6, 25.0);
    cov_pos[3] = std::clamp(cov_pos[3], 1e-6, 25.0);
    double max_cov = std::sqrt(cov_pos[0] * cov_pos[3]) * 0.99;
    sym01 = std::clamp(sym01, -max_cov, max_cov);
    cov_pos[1] = cov_pos[2] = sym01;

    time_now += dt_steps;
    age += 1;
    return get_bbox(false);
}

void VEMA3DTrack::mark_missed(double dt_steps) {
    k_lost += std::max(dt_steps, 1e-4);
    if (state == TrackState::Confirmed) {
        state = TrackState::Lost;
    }
}

std::array<double, 2> VEMA3DTrack::kalman_pos_update(
    const std::array<double, 2>& obs_3d,
    const std::array<double, 4>& cov_obs,
    bool allow_nis_boost) {
    std::array<double, 4> p_pred = cov_pos;
    std::array<double, 4> s_nom = {
        p_pred[0] + cov_obs[0],
        p_pred[1] + cov_obs[1],
        p_pred[2] + cov_obs[2],
        p_pred[3] + cov_obs[3],
    };
    std::array<double, 2> innov = {obs_3d[0] - pos_3d[0], obs_3d[1] - pos_3d[1]};
    double d2 = mahalanobis_sq_2x2(innov, s_nom);

    if (allow_nis_boost && d2 > 6.0) {
        consec_high_nis_ += 1;
    } else {
        consec_high_nis_ = 0;
    }

    std::array<double, 4> p_eff = p_pred;
    if (allow_nis_boost && consec_high_nis_ >= 2) {
        double boost = std::clamp(d2 / 4.0, 1.0, 16.0);
        for (double& v : p_eff) v *= boost;
    }

    std::array<double, 4> s_mat = {
        p_eff[0] + cov_obs[0],
        p_eff[1] + cov_obs[1],
        p_eff[2] + cov_obs[2],
        p_eff[3] + cov_obs[3],
    };
    double det = 0.0;
    std::array<double, 4> s_inv{};
    inv_2x2(s_mat, det, s_inv);

    // K = P_eff @ S_inv
    std::array<double, 4> k_gain = {
        p_eff[0] * s_inv[0] + p_eff[1] * s_inv[2],
        p_eff[0] * s_inv[1] + p_eff[1] * s_inv[3],
        p_eff[2] * s_inv[0] + p_eff[3] * s_inv[2],
        p_eff[2] * s_inv[1] + p_eff[3] * s_inv[3],
    };

    std::array<double, 2> pos_filt = {
        pos_3d[0] + k_gain[0] * innov[0] + k_gain[1] * innov[1],
        std::clamp(pos_3d[1] + k_gain[2] * innov[0] + k_gain[3] * innov[1], 0.3, camera_->max_depth),
    };

    // Joseph-form covariance update: (I - K) * P_eff * (I - K)^T + K * R * K^T
    std::array<double, 4> ik = {1.0 - k_gain[0], -k_gain[1], -k_gain[2], 1.0 - k_gain[3]};
    auto matmul = [](const std::array<double, 4>& a, const std::array<double, 4>& b) {
        return std::array<double, 4>{
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        };
    };
    auto transpose = [](const std::array<double, 4>& a) {
        return std::array<double, 4>{a[0], a[2], a[1], a[3]};
    };
    auto t1 = matmul(matmul(ik, p_eff), transpose(ik));
    auto t2 = matmul(matmul(k_gain, cov_obs), transpose(k_gain));
    std::array<double, 4> p_post = {t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2], t1[3] + t2[3]};
    double sym01 = 0.5 * (p_post[1] + p_post[2]);
    cov_pos = {std::max(p_post[0], 1e-6), sym01, sym01, std::max(p_post[3], 1e-6)};
    return pos_filt;
}

bool VEMA3DTrack::is_leg_occlusion(const BBox& obs_box, const BBox& pred_box) const {
    double pred_h = std::max(pred_box.y2 - pred_box.y1, 1.0);
    double obs_h = std::max(obs_box.y2 - obs_box.y1, 1e-3);
    if (obs_h / pred_h >= cfg_->area_ratio_min) {
        return false;
    }
    double dy1 = std::abs(obs_box.y1 - pred_box.y1);
    double dy2_up = pred_box.y2 - obs_box.y2;
    return dy1 <= std::max(8.0, 0.18 * pred_h) && dy2_up >= std::max(6.0, 0.12 * pred_h) && dy1 < dy2_up;
}

void VEMA3DTrack::update(const BBox& obs_box, double obs_conf, int64_t new_cls_id, int det_idx) {
    conf = obs_conf;
    cls_id = new_cls_id;
    associated_det_idx = det_idx;

    double elapsed = std::max(time_now - last_obs_time, 1e-6);
    double alpha_eff = std::pow(cfg_->alpha, std::min(elapsed, 5.0));
    auto c2d = obs_cov_2d(obs_box, obs_conf);

    if (obs_conf >= cfg_->tau_high) {
        std::array<double, 2> obs_3d{};
        std::array<double, 4> cov_obs{};
        double obs_pole_h = 0.0, obs_pole_w = 0.0, obs_elev = 0.0;
        camera_->bbox_to_pole_elev(obs_box, c2d, obs_3d, obs_pole_h, obs_pole_w, cov_obs, obs_elev);

        pos_3d = kalman_pos_update(obs_3d, cov_obs, true);
        std::array<double, 2> v_obs_3d = {
            (pos_3d[0] - last_obs_3d[0]) / elapsed,
            (pos_3d[1] - last_obs_3d[1]) / elapsed,
        };
        if (!has_vel) {
            v_3d = v_obs_3d;
            has_vel = true;
        } else {
            double dx = v_obs_3d[0] - v_3d[0];
            double dz = v_obs_3d[1] - v_3d[1];
            double err_sq = dx * dx + dz * dz;
            double var_alpha = std::pow(cfg_->v_var_alpha, std::min(elapsed, 5.0));
            v_var_3d = var_alpha * v_var_3d + (1.0 - var_alpha) * err_sq;
            v_3d[0] = alpha_eff * v_3d[0] + (1.0 - alpha_eff) * v_obs_3d[0];
            v_3d[1] = alpha_eff * v_3d[1] + (1.0 - alpha_eff) * v_obs_3d[1];
        }

        double size_alpha = std::min(alpha_eff, 0.55);
        pole_h = std::max(size_alpha * pole_h + (1.0 - size_alpha) * obs_pole_h, 0.05);
        pole_w = std::max(size_alpha * pole_w + (1.0 - size_alpha) * obs_pole_w, 0.02);
        base_elev = size_alpha * base_elev + (1.0 - size_alpha) * obs_elev;

        k_lost = 0.0;
        k_degraded = 0;
        last_anchor_head = false;
        last_obs_3d = pos_3d;
        last_obs_bbox = obs_box;
        last_obs_time = time_now;
        last_beta = 1.0;
        hits += 1;
        if (hits >= cfg_->probation_frames) {
            state = TrackState::Confirmed;
        }
        return;
    }

    if (obs_conf >= cfg_->tau_low) {
        BBox pred_box = get_bbox(false);
        double pred_w = std::max(pred_box.x2 - pred_box.x1, 1.0);
        double pred_h = std::max(pred_box.y2 - pred_box.y1, 1.0);
        double obs_w = std::max(obs_box.x2 - obs_box.x1, 1e-3);
        double obs_h = std::max(obs_box.y2 - obs_box.y1, 1e-3);

        double min_speed_3d = std::max(0.012, 0.025 * std::max(pole_w, 0.3));
        double norm_v_ema = std::hypot(v_3d[0], v_3d[1]);

        if (cfg_->enable_truncation_gate && is_leg_occlusion(obs_box, pred_box)) {
            last_anchor_head = true;
            std::array<double, 2> obs_head_3d{};
            std::array<double, 4> cov_head{};
            camera_->head_bbox_to_pole(
                obs_box, pole_h + base_elev, c2d, pos_3d[1], base_elev, obs_head_3d, cov_head);
            auto pos_head_filt = kalman_pos_update(obs_head_3d, cov_head, false);
            std::array<double, 2> v_obs_head = {
                (pos_head_filt[0] - last_obs_3d[0]) / elapsed,
                (pos_head_filt[1] - last_obs_3d[1]) / elapsed,
            };
            double norm_obs = std::hypot(v_obs_head[0], v_obs_head[1]);
            double cos_sim = 1.0;
            if (has_vel && norm_v_ema > min_speed_3d && norm_obs > min_speed_3d) {
                cos_sim = (v_obs_head[0] * v_3d[0] + v_obs_head[1] * v_3d[1]) / (norm_obs * norm_v_ema);
            }

            if (cos_sim < cfg_->cos_min) {
                last_beta = 0.0;
                k_degraded += 1;
            } else {
                double beta = cfg_->enable_velocity_blending
                                  ? std::clamp((obs_conf - cfg_->tau_low) / (cfg_->tau_high - cfg_->tau_low), 0.0, 1.0)
                                  : 1.0;
                last_beta = beta;
                if (!has_vel) {
                    v_3d = {beta * v_obs_head[0], beta * v_obs_head[1]};
                    has_vel = true;
                } else {
                    double max_delta = cfg_->eps_cal * (norm_v_ema + 0.05);
                    double dv0 = std::clamp(beta * (v_obs_head[0] - v_3d[0]), -max_delta, max_delta);
                    double dv1 = std::clamp(beta * (v_obs_head[1] - v_3d[1]), -max_delta, max_delta);
                    v_3d[0] += dv0;
                    v_3d[1] += dv1;
                }
                pos_3d = pos_head_filt;
                k_lost = 0.0;
                k_degraded = 0;
                last_obs_3d = pos_3d;
                last_obs_bbox = obs_box;
                last_obs_time = time_now;
                hits += 1;
            }

            if (cfg_->enable_size_lock) {
                pole_h = pre_predict_pole_h_;
                pole_w = pre_predict_pole_w_;
                base_elev = pre_predict_elev_;
            }
            if (state == TrackState::Lost) {
                state = TrackState::Confirmed;
            }
            return;
        }

        std::array<double, 2> obs_3d{};
        std::array<double, 4> cov_obs{};
        double obs_pole_h = 0.0, obs_pole_w = 0.0, obs_elev = 0.0;
        camera_->bbox_to_pole_elev(obs_box, c2d, obs_3d, obs_pole_h, obs_pole_w, cov_obs, obs_elev);

        double area_ratio = (obs_w * obs_h) / std::max(pred_w * pred_h, 1e-6);
        std::array<double, 2> v_obs_raw = {
            (obs_3d[0] - last_obs_3d[0]) / elapsed,
            (obs_3d[1] - last_obs_3d[1]) / elapsed,
        };
        double norm_obs_raw = std::hypot(v_obs_raw[0], v_obs_raw[1]);
        double cos_sim = 1.0;
        if (has_vel && norm_v_ema > min_speed_3d && norm_obs_raw > min_speed_3d) {
            cos_sim = (v_obs_raw[0] * v_3d[0] + v_obs_raw[1] * v_3d[1]) / (norm_obs_raw * norm_v_ema);
        }

        bool gated = cfg_->enable_truncation_gate && (area_ratio < cfg_->area_ratio_min || cos_sim < cfg_->cos_min);
        double beta = 0.0;
        if (gated) {
            beta = 0.0;
        } else if (!cfg_->enable_velocity_blending) {
            beta = 1.0;
        } else {
            beta = std::clamp((obs_conf - cfg_->tau_low) / (cfg_->tau_high - cfg_->tau_low), 0.0, 1.0);
        }

        last_beta = beta;
        if (beta > 0.0) {
            auto pos_filt = kalman_pos_update(obs_3d, cov_obs, false);
            std::array<double, 2> v_obs_filt = {
                (pos_filt[0] - last_obs_3d[0]) / elapsed,
                (pos_filt[1] - last_obs_3d[1]) / elapsed,
            };
            std::array<double, 2> v_blend{};
            if (!has_vel) {
                v_blend = {beta * v_obs_filt[0], beta * v_obs_filt[1]};
                v_3d = v_blend;
                has_vel = true;
            } else {
                v_blend = {
                    beta * v_obs_filt[0] + (1.0 - beta) * v_3d[0],
                    beta * v_obs_filt[1] + (1.0 - beta) * v_3d[1],
                };
                double max_delta = cfg_->eps_cal * (norm_v_ema + 0.05);
                double dv0 = std::clamp(beta * (v_obs_filt[0] - v_3d[0]), -max_delta, max_delta);
                double dv1 = std::clamp(beta * (v_obs_filt[1] - v_3d[1]), -max_delta, max_delta);
                v_3d[0] += dv0;
                v_3d[1] += dv1;
            }
            pos_3d[0] = last_obs_3d[0] + v_blend[0] * elapsed;
            pos_3d[1] = std::clamp(last_obs_3d[1] + v_blend[1] * elapsed, 0.3, camera_->max_depth);
            k_lost = 0.0;
            k_degraded = 0;
            last_anchor_head = false;
            last_obs_3d = pos_3d;
            last_obs_bbox = obs_box;
            last_obs_time = time_now;
            hits += 1;
        } else {
            if (cfg_->enable_size_lock) {
                pole_h = pre_predict_pole_h_;
                pole_w = pre_predict_pole_w_;
                base_elev = pre_predict_elev_;
            }
            k_degraded += 1;
        }

        const bool expanding = (beta > 0.0 && obs_pole_h > pole_h);
        if (!cfg_->enable_size_lock || expanding) {
            double size_alpha = expanding ? std::min(alpha_eff, 0.55) : alpha_eff;
            pole_h = std::max(size_alpha * pole_h + (1.0 - size_alpha) * obs_pole_h, 0.05);
            pole_w = std::max(size_alpha * pole_w + (1.0 - size_alpha) * obs_pole_w, 0.02);
            if (expanding) {
                base_elev = size_alpha * base_elev + (1.0 - size_alpha) * obs_elev;
            }
        }
        if (state == TrackState::Lost) {
            state = TrackState::Confirmed;
        }
        return;
    }

    last_beta = 0.0;
    mark_missed(0.0);
}

BBox VEMA3DTrack::get_bbox(bool buffered) const {
    BBox box = camera_->project_pole_to_bbox(pos_3d, pole_h, pole_w, base_elev);
    if (buffered && k_lost > 0.0) {
        double expansion = std::min(1.0 + cfg_->biou_rate * k_lost, cfg_->biou_max_expansion);
        if (expansion != 1.0) {
            double cx_b = 0.5 * (box.x1 + box.x2);
            double cy_b = 0.5 * (box.y1 + box.y2);
            double hw = 0.5 * (box.x2 - box.x1) * expansion;
            double hh = 0.5 * (box.y2 - box.y1) * expansion;
            return BBox{cx_b - hw, cy_b - hh, cx_b + hw, cy_b + hh};
        }
    }
    return box;
}

AssignmentResult linear_assignment(
    const std::vector<double>& cost_matrix,
    int n_rows,
    int n_cols,
    double cost_limit) {
    AssignmentResult res;
    if (n_rows == 0 || n_cols == 0) {
        for (int r = 0; r < n_rows; ++r) res.unmatched_rows.push_back(r);
        for (int c = 0; c < n_cols; ++c) res.unmatched_cols.push_back(c);
        return res;
    }

    double sentinel = std::max(cost_limit * 100.0 + 1e4, 1e6);
    bool transposed = false;
    int r_dim = n_rows;
    int c_dim = n_cols;
    if (r_dim > c_dim) {
        std::swap(r_dim, c_dim);
        transposed = true;
    }

    auto safe_at = [&](int i, int j) -> double {
        double raw = transposed ? cost_matrix[j * n_cols + i] : cost_matrix[i * n_cols + j];
        return std::isfinite(raw) ? raw : sentinel;
    };

    std::vector<double> u(r_dim + 1, 0.0);
    std::vector<double> v(c_dim + 1, 0.0);
    std::vector<int> p(c_dim + 1, 0);
    std::vector<int> way(c_dim + 1, 0);

    constexpr double kInf = 1e18;
    for (int i = 1; i <= r_dim; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(c_dim + 1, kInf);
        std::vector<char> used(c_dim + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= c_dim; ++j) {
                if (!used[j]) {
                    double cur = safe_at(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= c_dim; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<char> matched_r(n_rows, 0);
    std::vector<char> matched_c(n_cols, 0);
    std::vector<std::pair<int, int>> raw_pairs;
    raw_pairs.reserve(r_dim);
    for (int j = 1; j <= c_dim; ++j) {
        if (p[j] > 0) {
            int r = transposed ? (j - 1) : (p[j] - 1);
            int c = transposed ? (p[j] - 1) : (j - 1);
            raw_pairs.emplace_back(r, c);
        }
    }
    std::sort(raw_pairs.begin(), raw_pairs.end());

    for (const auto& rc : raw_pairs) {
        double raw_cost = cost_matrix[rc.first * n_cols + rc.second];
        if (std::isfinite(raw_cost) && raw_cost <= cost_limit) {
            res.matches.push_back(rc);
            matched_r[rc.first] = 1;
            matched_c[rc.second] = 1;
        }
    }
    for (int r = 0; r < n_rows; ++r) {
        if (!matched_r[r]) res.unmatched_rows.push_back(r);
    }
    for (int c = 0; c < n_cols; ++c) {
        if (!matched_c[c]) res.unmatched_cols.push_back(c);
    }
    return res;
}

VEMA3DTracker::VEMA3DTracker(const Config& cfg)
    : cfg_(cfg),
      camera_(
          static_cast<double>(cfg.img_width),
          static_cast<double>(cfg.img_height),
          cfg.cam_height,
          cfg.cam_pitch_deg,
          cfg.min_ray_y,
          cfg.max_depth) {}

void VEMA3DTracker::refresh_track_camera_ptrs() {
    for (auto& trk : tracks_) {
        trk.set_camera(&camera_);
    }
}

void VEMA3DTracker::reset(double fps, int img_width, int img_height) {
    if (fps > 0.0) {
        cfg_.ref_fps = fps;
    }
    if (img_width > 0 && img_height > 0) {
        cfg_.img_width = img_width;
        cfg_.img_height = img_height;
    }
    camera_ = PinholeGroundCamera(
        static_cast<double>(cfg_.img_width),
        static_cast<double>(cfg_.img_height),
        cfg_.cam_height,
        cfg_.cam_pitch_deg,
        cfg_.min_ray_y,
        cfg_.max_depth);
    tracks_.clear();
    next_id_ = 1;
    frame_idx_ = 0;
}

void VEMA3DTracker::set_image_size(int img_width, int img_height) {
    if (img_width <= 0 || img_height <= 0) return;
    if (cfg_.img_width == img_width && cfg_.img_height == img_height) return;
    cfg_.img_width = img_width;
    cfg_.img_height = img_height;
    if (tracks_.empty() && frame_idx_ == 0) {
        camera_ = PinholeGroundCamera(
            static_cast<double>(cfg_.img_width),
            static_cast<double>(cfg_.img_height),
            cfg_.cam_height,
            cfg_.cam_pitch_deg,
            cfg_.min_ray_y,
            cfg_.max_depth);
    }
}

std::vector<TrackOutput> VEMA3DTracker::update(
    const std::vector<BBox>& dets,
    const std::vector<double>& scores,
    const std::vector<int64_t>& cls_ids,
    double dt_sec) {
    frame_idx_ += 1;

    if (frame_idx_ == 1 && cfg_.auto_pitch && tracks_.empty() && !dets.empty()) {
        camera_ = camera_.auto_pitch_for_bboxes(dets);
        refresh_track_camera_ptrs();
    }

    double dt_steps = 1.0;
    if (cfg_.dt_aware && dt_sec > 0.0) {
        dt_steps = std::clamp(dt_sec * cfg_.ref_fps, 1e-4, 30.0);
    }

    for (auto& trk : tracks_) {
        trk.predict(dt_steps);
    }

    std::vector<size_t> active_indices;
    std::vector<size_t> lost_indices;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].state == TrackState::Lost) {
            lost_indices.push_back(i);
        } else {
            active_indices.push_back(i);
        }
    }

    std::vector<int> high_idx;
    std::vector<int> low_idx;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (scores[i] >= cfg_.tau_high) {
            high_idx.push_back(static_cast<int>(i));
        } else if (scores[i] >= cfg_.tau_low) {
            low_idx.push_back(static_cast<int>(i));
        }
    }

    std::vector<std::array<double, 2>> all_pos_3d;
    std::vector<std::array<double, 4>> all_cov_3d;
    if (!dets.empty()) {
        camera_.bbox_to_pole_batch(dets, scores, all_pos_3d, all_cov_3d);
    }

    auto build_stage12_cost = [&](const std::vector<size_t>& trk_indices,
                                  const std::vector<int>& det_indices,
                                  double iou_thresh,
                                  bool check_leg_occ) -> std::vector<double> {
        const int n = static_cast<int>(trk_indices.size());
        const int m = static_cast<int>(det_indices.size());
        std::vector<double> cost(n * m, 0.0);
        if (n == 0 || m == 0) return cost;

        std::vector<BBox> trk_boxes(n);
        for (int ti = 0; ti < n; ++ti) {
            trk_boxes[ti] = tracks_[trk_indices[ti]].get_bbox(false);
        }
        double w_mmd = std::clamp(cfg_.lambda_mmd, 0.0, 1.0);
        double mmd_denom = std::max(cfg_.mmd_gate, 1e-6);

        for (int ti = 0; ti < n; ++ti) {
            const auto& trk = tracks_[trk_indices[ti]];
            const BBox& tbox = trk_boxes[ti];
            double trk_h = std::max(tbox.y2 - tbox.y1, 1.0);

            for (int di = 0; di < m; ++di) {
                int g_di = det_indices[di];
                if (cfg_.check_class_match && trk.cls_id != cls_ids[g_di]) {
                    cost[ti * m + di] = std::numeric_limits<double>::infinity();
                    continue;
                }
                const BBox& dbox = dets[g_di];
                double iou = box_iou(tbox, dbox);
                if (iou < iou_thresh) {
                    cost[ti * m + di] = std::numeric_limits<double>::infinity();
                    continue;
                }

                double c_val = 1.0 - iou;
                if (w_mmd > 0.0) {
                    std::array<double, 2> diff = {
                        all_pos_3d[g_di][0] - trk.pos_3d[0],
                        all_pos_3d[g_di][1] - trk.pos_3d[1],
                    };
                    std::array<double, 4> s_mat = {
                        trk.cov_pos[0] + all_cov_3d[g_di][0],
                        trk.cov_pos[1] + all_cov_3d[g_di][1],
                        trk.cov_pos[2] + all_cov_3d[g_di][2],
                        trk.cov_pos[3] + all_cov_3d[g_di][3],
                    };
                    double d2 = std::max(mahalanobis_sq_2x2(diff, s_mat), 0.0);

                    if (check_leg_occ && cfg_.enable_truncation_gate) {
                        double det_h = std::max(dbox.y2 - dbox.y1, 1e-3);
                        double h_ratio = det_h / trk_h;
                        double dy1 = std::abs(dbox.y1 - tbox.y1);
                        double dy2_up = tbox.y2 - dbox.y2;
                        if (h_ratio < cfg_.area_ratio_min &&
                            dy1 <= std::max(8.0, 0.18 * trk_h) &&
                            dy2_up >= std::max(6.0, 0.12 * trk_h) &&
                            dy1 < dy2_up) {
                            double dw = std::max(dbox.x2 - dbox.x1, 1e-3);
                            double dh = std::max(dbox.y2 - dbox.y1, 1e-3);
                            double su = std::max(2.0, 0.008 * dw);
                            double sv = std::max(2.0, 0.006 * dh);
                            std::array<double, 4> c2d = {su * su, 0.0, 0.0, sv * sv};
                            std::array<double, 2> h_pos{};
                            std::array<double, 4> h_cov{};
                            camera_.head_bbox_to_pole(
                                dbox, trk.pole_h + trk.base_elev, c2d, trk.pos_3d[1], trk.base_elev, h_pos, h_cov);
                            std::array<double, 2> h_diff = {h_pos[0] - trk.pos_3d[0], h_pos[1] - trk.pos_3d[1]};
                            std::array<double, 4> h_s = {
                                trk.cov_pos[0] + h_cov[0],
                                trk.cov_pos[1] + h_cov[1],
                                trk.cov_pos[2] + h_cov[2],
                                trk.cov_pos[3] + h_cov[3],
                            };
                            double d2_head = mahalanobis_sq_2x2(h_diff, h_s);
                            if (d2_head < d2) {
                                d2 = d2_head;
                            }
                        }
                    }

                    double c_mmd = std::clamp(d2 / mmd_denom, 0.0, 1.0);
                    c_val = (1.0 - w_mmd) * (1.0 - iou) + w_mmd * c_mmd;
                }
                cost[ti * m + di] = c_val;
            }
        }
        return cost;
    };

    std::vector<char> matched_active(active_indices.size(), 0);
    std::vector<char> matched_lost(lost_indices.size(), 0);
    std::vector<int> unmatched_active_local;
    std::vector<int> unmatched_high_local;

    // Stage 1: Active <-> High-confidence
    if (!active_indices.empty() && !high_idx.empty()) {
        auto cost_s1 = build_stage12_cost(active_indices, high_idx, cfg_.iou_thresh_high, false);
        auto res1 = linear_assignment(
            cost_s1,
            static_cast<int>(active_indices.size()),
            static_cast<int>(high_idx.size()),
            1.0 - cfg_.iou_thresh_high);
        for (const auto& rc : res1.matches) {
            int g_di = high_idx[rc.second];
            tracks_[active_indices[rc.first]].update(dets[g_di], scores[g_di], cls_ids[g_di], g_di);
            matched_active[rc.first] = 1;
        }
        unmatched_active_local = std::move(res1.unmatched_rows);
        unmatched_high_local = std::move(res1.unmatched_cols);
    } else {
        for (size_t i = 0; i < active_indices.size(); ++i) unmatched_active_local.push_back(static_cast<int>(i));
        for (size_t j = 0; j < high_idx.size(); ++j) unmatched_high_local.push_back(static_cast<int>(j));
    }

    // Stage 2: Remaining Confirmed Active <-> Low-confidence
    std::vector<int> rem_confirmed_local;
    std::vector<size_t> rem_confirmed_trk_indices;
    for (int ti : unmatched_active_local) {
        if (tracks_[active_indices[ti]].state == TrackState::Confirmed) {
            rem_confirmed_local.push_back(ti);
            rem_confirmed_trk_indices.push_back(active_indices[ti]);
        }
    }
    if (!rem_confirmed_trk_indices.empty() && !low_idx.empty()) {
        auto cost_s2 = build_stage12_cost(rem_confirmed_trk_indices, low_idx, cfg_.iou_thresh_low, true);
        auto res2 = linear_assignment(
            cost_s2,
            static_cast<int>(rem_confirmed_trk_indices.size()),
            static_cast<int>(low_idx.size()),
            1.0 - cfg_.iou_thresh_low);
        for (const auto& rc : res2.matches) {
            int active_local_ti = rem_confirmed_local[rc.first];
            int g_di = low_idx[rc.second];
            tracks_[active_indices[active_local_ti]].update(dets[g_di], scores[g_di], cls_ids[g_di], g_di);
            matched_active[active_local_ti] = 1;
        }
    }

    // Stage 3: Lost <-> Remaining High-confidence
    std::vector<int> rem_high_global;
    rem_high_global.reserve(unmatched_high_local.size());
    for (int di : unmatched_high_local) {
        rem_high_global.push_back(high_idx[di]);
    }
    std::vector<int> unmatched_high_global = rem_high_global;

    if (!lost_indices.empty() && !rem_high_global.empty()) {
        const int n = static_cast<int>(lost_indices.size());
        const int m = static_cast<int>(rem_high_global.size());
        std::vector<double> cost_s3(n * m, 0.0);

        for (int li = 0; li < n; ++li) {
            const auto& trk = tracks_[lost_indices[li]];
            BBox trk_box = trk.get_bbox(cfg_.enable_biou);

            double min_dim_3d = std::max(trk.pole_w, 0.3);
            double base = std::max(min_dim_3d * cfg_.base_pos_sigma_ratio, 0.15);
            double k = std::max(trk.k_lost, 1.0);
            double v_std = std::sqrt(std::max(trk.v_var_3d, 0.0));
            double sigma_base = base + v_std * k;
            double norm_v = std::hypot(trk.v_3d[0], trk.v_3d[1]);
            double min_speed_3d = std::max(0.012, 0.025 * min_dim_3d);

            std::array<double, 4> s_motion{};
            bool use_aniso = cfg_.enable_anisotropic_cost && trk.has_vel && norm_v > min_speed_3d;
            if (use_aniso) {
                double ux = trk.v_3d[0] / norm_v;
                double uz = trk.v_3d[1] / norm_v;
                double px = -uz;
                double pz = ux;
                double sig_par = (sigma_base * cfg_.sigma_par_scale) * (sigma_base * cfg_.sigma_par_scale);
                double sig_perp = (sigma_base * cfg_.sigma_perp_scale) * (sigma_base * cfg_.sigma_perp_scale);
                s_motion = {
                    sig_par * ux * ux + sig_perp * px * px,
                    sig_par * ux * uz + sig_perp * px * pz,
                    sig_par * uz * ux + sig_perp * pz * px,
                    sig_par * uz * uz + sig_perp * pz * pz,
                };
            } else {
                double var = sigma_base * sigma_base;
                s_motion = {var, 0.0, 0.0, var};
            }

            for (int di = 0; di < m; ++di) {
                int g_di = rem_high_global[di];
                if (cfg_.check_class_match && trk.cls_id != cls_ids[g_di]) {
                    cost_s3[li * m + di] = std::numeric_limits<double>::infinity();
                    continue;
                }
                double biou = box_iou(trk_box, dets[g_di]);
                std::array<double, 2> e = {
                    all_pos_3d[g_di][0] - trk.pos_3d[0],
                    all_pos_3d[g_di][1] - trk.pos_3d[1],
                };
                std::array<double, 4> s_tot = {
                    s_motion[0] + all_cov_3d[g_di][0],
                    s_motion[1] + all_cov_3d[g_di][1],
                    s_motion[2] + all_cov_3d[g_di][2],
                    s_motion[3] + all_cov_3d[g_di][3],
                };
                double c_pos = std::clamp(mahalanobis_sq_2x2(e, s_tot), 0.0, 10.0);
                double c_val = cfg_.lambda_biou * (1.0 - biou) + cfg_.lambda_pos * c_pos;

                if (cfg_.enable_dir_cost && cfg_.lambda_dir > 0.0 && trk.has_vel && norm_v > min_speed_3d) {
                    double cx_d = all_pos_3d[g_di][0] - trk.last_obs_3d[0];
                    double cz_d = all_pos_3d[g_di][1] - trk.last_obs_3d[1];
                    double norm_c = std::hypot(cx_d, cz_d);
                    if (norm_c > min_speed_3d) {
                        double cos_t = std::clamp((cx_d * trk.v_3d[0] + cz_d * trk.v_3d[1]) / (norm_c * norm_v), -1.0, 1.0);
                        c_val += cfg_.lambda_dir * (1.0 - cos_t);
                    }
                }
                cost_s3[li * m + di] = c_val;
            }
        }

        auto res3 = linear_assignment(cost_s3, n, m, cfg_.relocation_cost_thresh);
        for (const auto& rc : res3.matches) {
            int g_di = rem_high_global[rc.second];
            tracks_[lost_indices[rc.first]].update(dets[g_di], scores[g_di], cls_ids[g_di], g_di);
            matched_lost[rc.first] = 1;
        }
        unmatched_high_global.clear();
        for (int di : res3.unmatched_cols) {
            unmatched_high_global.push_back(rem_high_global[di]);
        }
    }

    for (size_t ti = 0; ti < active_indices.size(); ++ti) {
        if (!matched_active[ti]) {
            tracks_[active_indices[ti]].mark_missed(dt_steps);
        }
    }
    for (size_t li = 0; li < lost_indices.size(); ++li) {
        if (!matched_lost[li]) {
            tracks_[lost_indices[li]].mark_missed(dt_steps);
        }
    }

    std::vector<VEMA3DTrack> kept_tracks;
    kept_tracks.reserve(tracks_.size());
    const double img_w = static_cast<double>(cfg_.img_width);
    const double img_h = static_cast<double>(cfg_.img_height);
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Tentative && trk.k_lost > static_cast<double>(cfg_.early_term_lost)) {
            continue;
        }
        if (trk.k_lost > static_cast<double>(cfg_.max_lost_frames)) {
            continue;
        }
        if (trk.state == TrackState::Lost && cfg_.border_exit_lost > 0 && img_w > 0.0 && img_h > 0.0) {
            const BBox pbox = trk.get_bbox(false);
            const double area = std::max(pbox.x2 - pbox.x1, 0.0) * std::max(pbox.y2 - pbox.y1, 0.0);
            const double vis_w = std::max(0.0, std::min(pbox.x2, img_w) - std::max(pbox.x1, 0.0));
            const double vis_h = std::max(0.0, std::min(pbox.y2, img_h) - std::max(pbox.y1, 0.0));
            const double vis_frac = (area > 1e-6) ? ((vis_w * vis_h) / area) : 0.0;
            const bool mostly_offscreen = (vis_frac < cfg_.min_visible_frac) && (trk.k_lost > 1.0);
            const double m = cfg_.border_exit_margin_px;
            const bool at_border =
                (pbox.x1 <= m) || (pbox.y1 <= m) || (pbox.x2 >= img_w - m) || (pbox.y2 >= img_h - m);
            if (mostly_offscreen || (at_border && trk.k_lost > static_cast<double>(cfg_.border_exit_lost))) {
                continue;
            }
        }
        kept_tracks.push_back(std::move(trk));
    }
    tracks_ = std::move(kept_tracks);

    if (!unmatched_high_global.empty()) {
        std::vector<BBox> existing_boxes;
        for (const auto& trk : tracks_) {
            if (trk.state != TrackState::Lost) {
                existing_boxes.push_back(trk.get_bbox(false));
            }
        }
        for (int g_di : unmatched_high_global) {
            const BBox& det_box = dets[g_di];
            double max_iou = 0.0;
            for (const auto& eb : existing_boxes) {
                max_iou = std::max(max_iou, box_iou(det_box, eb));
            }
            if (!existing_boxes.empty() && max_iou > cfg_.min_iou_diff_new) {
                continue;
            }
            VEMA3DTrack trk(next_id_++, det_box, scores[g_di], cls_ids[g_di], cfg_, camera_, g_di);
            if (frame_idx_ == 1) {
                trk.state = TrackState::Confirmed;
            }
            tracks_.push_back(std::move(trk));
        }
    }
    refresh_track_camera_ptrs();

    std::vector<TrackOutput> outputs;
    outputs.reserve(tracks_.size());
    for (const auto& trk : tracks_) {
        if (trk.state == TrackState::Confirmed) {
            outputs.push_back(TrackOutput{
                trk.id, trk.get_bbox(false), trk.conf, trk.cls_id, false, trk.age, trk.associated_det_idx});
        } else if (cfg_.emit_shadow_tracks && trk.state == TrackState::Lost) {
            if (cfg_.max_shadow_emit_lost <= 0 ||
                trk.k_lost <= static_cast<double>(cfg_.max_shadow_emit_lost)) {
                BBox out_box = trk.get_bbox(false);
                if (cfg_.border_exit_lost > 0 && img_w > 0.0 && img_h > 0.0) {
                    const double m = cfg_.border_exit_margin_px;
                    const bool at_border = (out_box.x1 <= m) || (out_box.y1 <= m) ||
                                           (out_box.x2 >= img_w - m) || (out_box.y2 >= img_h - m);
                    if (at_border) {
                        continue;
                    }
                }
                outputs.push_back(TrackOutput{
                    trk.id, out_box, trk.conf, trk.cls_id, true, trk.age, -1});
            }
        }
    }
    return outputs;
}

}  // namespace vema3d

namespace {

struct TestHarness {
    vema3d::Config cfg;
    std::unordered_map<uint64_t, vema3d::VEMA3DTracker> streams;
};

}  // namespace

extern "C" {

void* vema3d_test_create(const char* yaml_path, int img_width, int img_height) {
    auto* h = new TestHarness();
    if (yaml_path != nullptr && yaml_path[0] != '\0') {
        h->cfg = vema3d::Config::from_yaml_file(yaml_path);
    }
    if (img_width > 0 && img_height > 0) {
        h->cfg.img_width = img_width;
        h->cfg.img_height = img_height;
    }
    return static_cast<void*>(h);
}

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
    int max_out) {
    if (handle == nullptr || num_dets < 0 || max_out < 0) {
        return -1;
    }
    auto* h = static_cast<TestHarness*>(handle);
    auto it = h->streams.find(stream_id);
    if (it == h->streams.end()) {
        it = h->streams.emplace(stream_id, vema3d::VEMA3DTracker(h->cfg)).first;
    }

    std::vector<vema3d::BBox> dets(static_cast<size_t>(num_dets));
    std::vector<double> sc(static_cast<size_t>(num_dets));
    std::vector<int64_t> cls(static_cast<size_t>(num_dets), 0);
    for (int i = 0; i < num_dets; ++i) {
        dets[i] = vema3d::BBox{
            dets_xyxy[4 * i + 0],
            dets_xyxy[4 * i + 1],
            dets_xyxy[4 * i + 2],
            dets_xyxy[4 * i + 3],
        };
        sc[i] = scores[i];
        if (cls_ids != nullptr) {
            cls[i] = cls_ids[i];
        }
    }

    auto outs = it->second.update(dets, sc, cls, dt_sec);
    int n_emit = std::min(static_cast<int>(outs.size()), max_out);
    for (int i = 0; i < n_emit; ++i) {
        out_ids[i] = outs[i].track_id;
        out_boxes_xyxy[4 * i + 0] = outs[i].bbox.x1;
        out_boxes_xyxy[4 * i + 1] = outs[i].bbox.y1;
        out_boxes_xyxy[4 * i + 2] = outs[i].bbox.x2;
        out_boxes_xyxy[4 * i + 3] = outs[i].bbox.y2;
        out_confs[i] = outs[i].conf;
        out_cls_ids[i] = outs[i].cls_id;
        out_is_shadow[i] = outs[i].is_shadow ? 1 : 0;
    }
    return static_cast<int>(outs.size());
}

int vema3d_test_remove_stream(void* handle, uint64_t stream_id) {
    if (handle == nullptr) return -1;
    auto* h = static_cast<TestHarness*>(handle);
    return static_cast<int>(h->streams.erase(stream_id));
}

void vema3d_test_destroy(void* handle) {
    delete static_cast<TestHarness*>(handle);
}

}  // extern "C"
