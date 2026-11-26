#include "vema2_tracker.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

namespace vema2 {

namespace {

constexpr double kEps = 1e-8;

double clamp_val(double x, double lo, double hi) {
    return std::max(lo, std::min(x, hi));
}

void box_to_cxcywh(const BBox& b, double& cx, double& cy, double& w, double& h) {
    cx = 0.5 * (b.x1 + b.x2);
    cy = 0.5 * (b.y1 + b.y2);
    w = std::max(b.x2 - b.x1, 1.0);
    h = std::max(b.y2 - b.y1, 1.0);
}

double compute_iou(const BBox& a, const BBox& b) {
    const double x1 = std::max(a.x1, b.x1);
    const double y1 = std::max(a.y1, b.y1);
    const double x2 = std::min(a.x2, b.x2);
    const double y2 = std::min(a.y2, b.y2);
    const double iw = std::max(0.0, x2 - x1);
    const double ih = std::max(0.0, y2 - y1);
    const double inter = iw * ih;
    const double area_a = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
    const double area_b = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
    const double uni = area_a + area_b - inter;
    return (uni > 1e-9) ? (inter / uni) : 0.0;
}

std::string trim_copy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string strip_comment(const std::string& s) {
    bool in_quote = false;
    char qchar = '\0';
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (in_quote) {
            if (c == qchar) {
                in_quote = false;
            }
        } else {
            if (c == '\'' || c == '"') {
                in_quote = true;
                qchar = c;
            } else if (c == '#') {
                return s.substr(0, i);
            }
        }
    }
    return s;
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool parse_bool_val(const std::string& val, bool fallback) {
    std::string low = unquote(trim_copy(val));
    for (char& c : low) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (low == "true" || low == "1" || low == "yes" || low == "on") return true;
    if (low == "false" || low == "0" || low == "no" || low == "off") return false;
    return fallback;
}

double parse_double_val(const std::string& val, double fallback) {
    try {
        std::string s = unquote(trim_copy(val));
        size_t pos = 0;
        double out = std::stod(s, &pos);
        if (pos == 0 || !std::isfinite(out)) return fallback;
        return out;
    } catch (...) {
        return fallback;
    }
}

int parse_int_val(const std::string& val, int fallback) {
    try {
        std::string s = unquote(trim_copy(val));
        size_t pos = 0;
        int out = std::stoi(s, &pos);
        if (pos == 0) return fallback;
        return out;
    } catch (...) {
        return fallback;
    }
}

// Exact O(n^3) shortest-augmenting-path Hungarian solver for rectangular cost matrices.
void hungarian_solve(
    const std::vector<double>& cost,
    int n_rows,
    int n_cols,
    std::vector<int>& out_rows,
    std::vector<int>& out_cols) {
    out_rows.clear();
    out_cols.clear();
    if (n_rows <= 0 || n_cols <= 0) {
        return;
    }

    const bool transposed = n_rows > n_cols;
    const int nr = transposed ? n_cols : n_rows;
    const int nc = transposed ? n_rows : n_cols;

    auto get_cost = [&](int r, int c) -> double {
        return transposed ? cost[c * n_cols + r] : cost[r * n_cols + c];
    };

    const double inf = 1e18;
    std::vector<double> u(nr + 1, 0.0);
    std::vector<double> v(nc + 1, 0.0);
    std::vector<int> p(nc + 1, 0);
    std::vector<int> way(nc + 1, 0);

    for (int i = 1; i <= nr; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(nc + 1, inf);
        std::vector<bool> used(nc + 1, false);
        while (true) {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= nc; ++j) {
                if (!used[j]) {
                    const double cur = get_cost(i0 - 1, j - 1) - u[i0] - v[j];
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
            for (int j = 0; j <= nc; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) {
                break;
            }
        }
        while (true) {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) {
                break;
            }
        }
    }

    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(nr);
    for (int j = 1; j <= nc; ++j) {
        if (p[j] > 0) {
            const int r = p[j] - 1;
            const int c = j - 1;
            if (transposed) {
                pairs.emplace_back(c, r);
            } else {
                pairs.emplace_back(r, c);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    for (const auto& pr : pairs) {
        out_rows.push_back(pr.first);
        out_cols.push_back(pr.second);
    }
}

void compute_polar_kinematic_row(
    const PolarEMATrack& trk,
    const std::vector<BBox>& dets,
    const std::vector<int>& det_indices,
    double t_sec,
    double dt_sec,
    std::vector<double>& out_c_kin,
    double& out_rho,
    double& out_noise_sec) {
    const size_t m = det_indices.size();
    out_c_kin.assign(m, 0.0);

    out_noise_sec = trk.sigma_noise_sec(dt_sec);
    out_rho = trk.rho(dt_sec);
    const double dt_a = std::max(t_sec - trk.t_anchor, 1e-4);
    const auto v_ema = trk.v_ema_sec();
    const double ema_norm = std::hypot(v_ema.first, v_ema.second);

    for (size_t j = 0; j < m; ++j) {
        const BBox& b = dets[static_cast<size_t>(det_indices[j])];
        const double cx = 0.5 * (b.x1 + b.x2);
        const double cy = 0.5 * (b.y1 + b.y2);
        const double vx = (cx - trk.p_anchor_x) / dt_a;
        const double vy = (cy - trk.p_anchor_y) / dt_a;
        const double cand_norm = std::hypot(vx, vy);
        const double diff_norm = std::hypot(vx - v_ema.first, vy - v_ema.second);
        out_c_kin[j] = diff_norm / (cand_norm + ema_norm + out_noise_sec + 1e-6);
    }
}

std::vector<double> iou_polar_cost_matrix(
    const std::vector<PolarEMATrack*>& trks,
    const std::vector<BBox>& dets,
    const std::vector<int64_t>& cls_ids,
    const std::vector<int>& det_indices,
    double t_sec,
    double dt_sec,
    const Config& cfg,
    double iou_thresh) {
    const int n = static_cast<int>(trks.size());
    const int m = static_cast<int>(det_indices.size());
    std::vector<double> cost(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
    if (n == 0 || m == 0) {
        return cost;
    }

    std::vector<double> c_kin;
    for (int i = 0; i < n; ++i) {
        const PolarEMATrack& trk = *trks[static_cast<size_t>(i)];
        const BBox tb = trk.get_bbox();
        double rho = 0.0;
        double noise_sec = 0.0;
        compute_polar_kinematic_row(trk, dets, det_indices, t_sec, dt_sec, c_kin, rho, noise_sec);
        (void)noise_sec;

        for (int j = 0; j < m; ++j) {
            const int g_di = det_indices[static_cast<size_t>(j)];
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j);
            if (cfg.check_class_match && trk.cls_id != cls_ids[static_cast<size_t>(g_di)]) {
                cost[idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            const double iou = compute_iou(tb, dets[static_cast<size_t>(g_di)]);
            if (iou < iou_thresh) {
                cost[idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            cost[idx] = (1.0 - iou) + cfg.iou_thresh_high * rho * c_kin[static_cast<size_t>(j)];
        }
    }
    return cost;
}

std::vector<double> relocation_polar_cost_matrix(
    const std::vector<PolarEMATrack*>& lost_trks,
    const std::vector<BBox>& dets,
    const std::vector<int64_t>& cls_ids,
    const std::vector<int>& det_indices,
    double t_sec,
    double dt_sec,
    const Config& cfg) {
    const int n = static_cast<int>(lost_trks.size());
    const int m = static_cast<int>(det_indices.size());
    std::vector<double> cost(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0);
    if (n == 0 || m == 0) {
        return cost;
    }

    std::vector<double> c_kin;
    for (int i = 0; i < n; ++i) {
        const PolarEMATrack& trk = *lost_trks[static_cast<size_t>(i)];
        double rho = 0.0;
        double noise_sec = 0.0;
        compute_polar_kinematic_row(trk, dets, det_indices, t_sec, dt_sec, c_kin, rho, noise_sec);

        const double r_val = trk.R();
        const auto u = trk.u_hat();
        const double sigma_base = 0.5 * std::sqrt(std::max(trk.d_w * trk.d_h, 1.0));
        const double t_lost_sec = trk.k_lost / cfg.ref_fps;
        const double ang_std = std::sqrt(2.0 * std::max(1.0 - r_val, 0.0));
        const double sigma_par = sigma_base + t_lost_sec * (trk.s_dev + noise_sec);
        const double sigma_perp = sigma_base + t_lost_sec * (trk.s_ema * ang_std + noise_sec);
        const bool has_dir = r_val > 1e-3;

        for (int j = 0; j < m; ++j) {
            const int g_di = det_indices[static_cast<size_t>(j)];
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(m) + static_cast<size_t>(j);
            if (cfg.check_class_match && trk.cls_id != cls_ids[static_cast<size_t>(g_di)]) {
                cost[idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            const BBox& db = dets[static_cast<size_t>(g_di)];
            const double ex = 0.5 * (db.x1 + db.x2) - trk.pos_x;
            const double ey = 0.5 * (db.y1 + db.y2) - trk.pos_y;

            double dist = 0.0;
            if (has_dir) {
                const double proj = ex * u.first + ey * u.second;
                const double e_par = std::abs(proj);
                const double perp_x = ex - proj * u.first;
                const double perp_y = ey - proj * u.second;
                const double e_perp = std::hypot(perp_x, perp_y);
                dist = std::hypot(e_par / (sigma_par + 1e-6), e_perp / (sigma_perp + 1e-6));
            } else {
                dist = std::hypot(ex, ey) / (sigma_base + t_lost_sec * noise_sec + 1e-6);
            }
            const double c_dist = std::min(1.0, dist);
            const double ck = c_kin[static_cast<size_t>(j)];
            const bool feasible = (c_dist <= (1.0 - cfg.iou_thresh_high)) && !(rho > 0.3 && ck > 0.5);
            if (!feasible) {
                cost[idx] = std::numeric_limits<double>::infinity();
                continue;
            }
            cost[idx] = c_dist + cfg.iou_thresh_high * rho * ck;
        }
    }
    return cost;
}

}  // namespace

void Config::validate() {
    tau_low = clamp_val(tau_low, 0.0, 0.99);
    tau_high = clamp_val(tau_high, 0.01, 1.0);
    new_track_thresh = clamp_val(new_track_thresh, 0.0, 1.0);
    if (tau_low >= tau_high) {
        std::fprintf(
            stderr,
            "[vema2] Invalid confidence thresholds tau_low=%.4f >= tau_high=%.4f; clamping tau_low.\n",
            tau_low,
            tau_high);
        tau_low = std::max(0.0, tau_high - 0.05);
    }
    if (!(tau_mem > 0.0) || !std::isfinite(tau_mem)) {
        std::fprintf(stderr, "[vema2] Invalid tau_mem; resetting to 0.15s.\n");
        tau_mem = 0.15;
    }
    if (!(ref_fps > 0.0) || !std::isfinite(ref_fps)) {
        ref_fps = 30.0;
    }
    probation_frames = std::max(1, probation_frames);
    max_lost_frames = std::max(1, max_lost_frames);
    early_term_lost = std::max(0, early_term_lost);
    max_targets_per_stream = std::max(1, max_targets_per_stream);
}

Config Config::from_yaml_string(const std::string& content) {
    Config cfg;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        std::string clean = trim_copy(strip_comment(line));
        if (clean.empty() || clean == "---" || clean.rfind("%YAML", 0) == 0) {
            continue;
        }
        const size_t colon = clean.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = trim_copy(clean.substr(0, colon));
        const std::string val = trim_copy(clean.substr(colon + 1));
        if (val.empty()) {
            continue;
        }

        if (key == "tau_high") cfg.tau_high = parse_double_val(val, cfg.tau_high);
        else if (key == "tau_low") cfg.tau_low = parse_double_val(val, cfg.tau_low);
        else if (key == "iou_thresh_high") cfg.iou_thresh_high = parse_double_val(val, cfg.iou_thresh_high);
        else if (key == "iou_thresh_low") cfg.iou_thresh_low = parse_double_val(val, cfg.iou_thresh_low);
        else if (key == "probation_frames") cfg.probation_frames = parse_int_val(val, cfg.probation_frames);
        else if (key == "max_lost_frames") cfg.max_lost_frames = parse_int_val(val, cfg.max_lost_frames);
        else if (key == "early_term_lost") cfg.early_term_lost = parse_int_val(val, cfg.early_term_lost);
        else if (key == "min_iou_diff_new") cfg.min_iou_diff_new = parse_double_val(val, cfg.min_iou_diff_new);
        else if (key == "check_class_match") cfg.check_class_match = parse_bool_val(val, cfg.check_class_match);
        else if (key == "emit_shadow_tracks") cfg.emit_shadow_tracks = parse_bool_val(val, cfg.emit_shadow_tracks);
        else if (key == "ref_fps") cfg.ref_fps = parse_double_val(val, cfg.ref_fps);
        else if (key == "dt_aware") cfg.dt_aware = parse_bool_val(val, cfg.dt_aware);
        else if (key == "tau_mem") cfg.tau_mem = parse_double_val(val, cfg.tau_mem);
        else if (key == "new_track_thresh") cfg.new_track_thresh = parse_double_val(val, cfg.new_track_thresh);
        else if (key == "border_exit_lost") cfg.border_exit_lost = parse_int_val(val, cfg.border_exit_lost);
        else if (key == "border_exit_margin_px") cfg.border_exit_margin_px = parse_double_val(val, cfg.border_exit_margin_px);
        else if (key == "min_visible_frac") cfg.min_visible_frac = parse_double_val(val, cfg.min_visible_frac);
        else if (key == "max_shadow_emit_lost") cfg.max_shadow_emit_lost = parse_int_val(val, cfg.max_shadow_emit_lost);
        else if (key == "max_targets_per_stream") cfg.max_targets_per_stream = parse_int_val(val, cfg.max_targets_per_stream);
        else if (key == "img_width") cfg.img_width = parse_int_val(val, cfg.img_width);
        else if (key == "img_height") cfg.img_height = parse_int_val(val, cfg.img_height);
    }
    cfg.validate();
    return cfg;
}

Config Config::from_yaml_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "[vema2] Could not open YAML config '%s'; using defaults.\n", path.c_str());
        Config cfg;
        cfg.validate();
        return cfg;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return from_yaml_string(oss.str());
}

AssignmentResult linear_assignment(
    const std::vector<double>& cost_matrix,
    int n_rows,
    int n_cols,
    double cost_limit) {
    AssignmentResult res;
    if (n_rows <= 0 || n_cols <= 0) {
        for (int r = 0; r < n_rows; ++r) res.unmatched_rows.push_back(r);
        for (int c = 0; c < n_cols; ++c) res.unmatched_cols.push_back(c);
        return res;
    }

    const double sentinel = std::max(cost_limit * 100.0 + 1e4, 1e6);
    std::vector<double> safe_cost(cost_matrix.size(), sentinel);
    for (size_t i = 0; i < cost_matrix.size(); ++i) {
        const double v = cost_matrix[i];
        // Pre-mask both non-finite entries and finite entries > cost_limit before Hungarian
        if (std::isfinite(v) && v <= cost_limit) {
            safe_cost[i] = v;
        }
    }

    std::vector<int> row_ind;
    std::vector<int> col_ind;
    hungarian_solve(safe_cost, n_rows, n_cols, row_ind, col_ind);

    std::vector<bool> row_matched(static_cast<size_t>(n_rows), false);
    std::vector<bool> col_matched(static_cast<size_t>(n_cols), false);

    for (size_t k = 0; k < row_ind.size(); ++k) {
        const int r = row_ind[k];
        const int c = col_ind[k];
        const double orig = cost_matrix[static_cast<size_t>(r) * static_cast<size_t>(n_cols) + static_cast<size_t>(c)];
        if (std::isfinite(orig) && orig <= cost_limit) {
            res.matches.emplace_back(r, c);
            row_matched[static_cast<size_t>(r)] = true;
            col_matched[static_cast<size_t>(c)] = true;
        }
    }
    for (int r = 0; r < n_rows; ++r) {
        if (!row_matched[static_cast<size_t>(r)]) res.unmatched_rows.push_back(r);
    }
    for (int c = 0; c < n_cols; ++c) {
        if (!col_matched[static_cast<size_t>(c)]) res.unmatched_cols.push_back(c);
    }
    return res;
}

PolarEMATrack::PolarEMATrack(
    uint64_t track_id,
    const BBox& bbox,
    double conf_in,
    int64_t cls_in,
    double t_sec,
    const Config& cfg)
    : id(track_id), cls_id(cls_in), conf(conf_in), t_anchor(t_sec) {
    box_to_cxcywh(bbox, pos_x, pos_y, d_w, d_h);
    p_anchor_x = pos_x;
    p_anchor_y = pos_y;
    const double denom = std::max(cfg.effective_new_track_thresh() - cfg.tau_low, 1e-6);
    health = clamp_val((conf_in - cfg.tau_low) / denom, 0.0, 1.0);
    const double s0 = 0.05 * std::min(d_w, d_h);
    p_var = s0 * s0;
}

double PolarEMATrack::R() const {
    return std::min(1.0, std::hypot(m_x, m_y));
}

std::pair<double, double> PolarEMATrack::u_hat() const {
    const double r = std::hypot(m_x, m_y);
    if (r <= 1e-6) {
        return {0.0, 0.0};
    }
    return {m_x / r, m_y / r};
}

std::pair<double, double> PolarEMATrack::v_ema_sec() const {
    return {s_ema * m_x, s_ema * m_y};
}

double PolarEMATrack::sigma_noise_px() const {
    const double raw = std::sqrt(std::max(inn_sq, 0.0));
    return std::min(raw, 0.10 * std::min(d_w, d_h));
}

double PolarEMATrack::sigma_noise_sec(double dt_sec) const {
    const double px_floor = std::max(sigma_noise_px(), 0.015 * std::min(d_w, d_h));
    return px_floor / std::max(dt_sec, 1e-3);
}

double PolarEMATrack::rho(double dt_sec) const {
    const double s_noise = sigma_noise_sec(dt_sec);
    const double s2 = s_ema * s_ema;
    return R() * s2 / (s2 + s_noise * s_noise + kEps);
}

BBox PolarEMATrack::predict(double dt_sec, const Config& cfg) {
    const double t_max = std::max(static_cast<double>(cfg.max_lost_frames) / cfg.ref_fps, 1e-6);
    const double t_lost_sec = k_lost / cfg.ref_fps;
    const double damping = std::exp(-t_lost_sec / t_max);
    const auto v = v_ema_sec();
    pos_x += damping * v.first * dt_sec;
    pos_y += damping * v.second * dt_sec;

    const double min_dim = std::min(d_w, d_h);
    const double sigma_q = (0.15 * min_dim) * (dt_sec / cfg.tau_mem) + s_dev * dt_sec;
    const double max_p = (0.5 * min_dim) * (0.5 * min_dim);
    p_var = std::min(p_var + sigma_q * sigma_q, max_p);
    ++age;
    return get_bbox();
}

void PolarEMATrack::mark_missed(double dt_sec, const Config& cfg) {
    const double alpha_h = 1.0 - std::exp(-dt_sec / cfg.tau_mem);
    health *= (1.0 - alpha_h);
    k_lost += dt_sec * cfg.ref_fps;
}

void PolarEMATrack::update(
    const BBox& obs_box,
    double obs_conf,
    int64_t obs_cls,
    double t_sec,
    double dt_sec,
    double max_other_iou,
    const Config& cfg) {
    double obs_cx = 0.0, obs_cy = 0.0, obs_w = 1.0, obs_h = 1.0;
    box_to_cxcywh(obs_box, obs_cx, obs_cy, obs_w, obs_h);

    const double denom = std::max(cfg.effective_new_track_thresh() - cfg.tau_low, 1e-6);
    const double w_conf = clamp_val((obs_conf - cfg.tau_low) / denom, 0.0, 1.0);
    const double w_occ = clamp_val(1.0 - max_other_iou, 0.0, 1.0);
    const double area_ratio = (obs_w * obs_h) / (d_w * d_h + 1e-6);
    const double w_size_lin = std::min(1.0, area_ratio);
    const double w_size = w_size_lin * w_size_lin;
    const double w_ema = w_conf * w_occ * w_size;

    const double dt_anchor = std::max(t_sec - t_anchor, 1e-4);
    const double v_obs_x = (obs_cx - p_anchor_x) / dt_anchor;
    const double v_obs_y = (obs_cy - p_anchor_y) / dt_anchor;
    const double s_obs = std::hypot(v_obs_x, v_obs_y);

    const double rx = obs_cx - pos_x;
    const double ry = obs_cy - pos_y;

    const double dt_mem_eff = std::min(dt_anchor, 5.0 * cfg.tau_mem);
    const double eta_v_base = 1.0 - std::exp(-dt_mem_eff / cfg.tau_mem);
    const double eta_d_base = eta_v_base;
    const double alpha_h = 1.0 - std::exp(-dt_sec / cfg.tau_mem);

    if (has_vel && k_lost == 0.0) {
        const double eta_inn = w_ema * eta_v_base;
        inn_sq = (1.0 - eta_inn) * inn_sq + eta_inn * (rx * rx + ry * ry);
    }

    // Scalar NSA Kalman gain + NIS maneuver boost
    const double sigma_meas = std::max(sigma_noise_px(), 0.02 * std::min(d_w, d_h));
    const double r_base = sigma_meas * sigma_meas;
    const double r_t_var = r_base / (w_conf + 1e-3);
    const double nis = (rx * rx + ry * ry) / (2.0 * (p_var + r_t_var) + kEps);
    const double p_eff = p_var * std::max(1.0, nis);
    const double kappa_t = p_eff / (p_eff + r_t_var + kEps);
    pos_x = (1.0 - kappa_t) * pos_x + kappa_t * obs_cx;
    pos_y = (1.0 - kappa_t) * pos_y + kappa_t * obs_cy;
    p_var = (1.0 - kappa_t) * p_eff;

    // Speed reliability gate using derived detector noise floor
    const double s_noise = sigma_noise_sec(dt_anchor);
    const double g_s = (s_obs * s_obs) / (s_obs * s_obs + s_noise * s_noise + kEps);
    const double ux = (s_obs > 1e-6) ? (v_obs_x / (s_obs + kEps)) : 0.0;
    const double uy = (s_obs > 1e-6) ? (v_obs_y / (s_obs + kEps)) : 0.0;

    if (!has_vel) {
        if (w_ema > 1e-3) {
            s_ema = s_obs;
            m_x = g_s * ux;
            m_y = g_s * uy;
            has_vel = true;
        }
    } else {
        const double eta_s = w_ema * eta_v_base;
        const double eta_u = w_ema * g_s * eta_v_base;
        const double eta_sd = w_ema * eta_d_base;
        s_dev = (1.0 - eta_sd) * s_dev + eta_sd * std::abs(s_obs - s_ema);
        s_ema = (1.0 - eta_s) * s_ema + eta_s * s_obs;
        m_x = (1.0 - eta_u) * m_x + eta_u * ux;
        m_y = (1.0 - eta_u) * m_y + eta_u * uy;
        const double m_norm = std::hypot(m_x, m_y);
        if (m_norm > 1.0) {
            m_x /= m_norm;
            m_y /= m_norm;
        }
    }

    const double eta_d = w_ema * eta_d_base;
    d_w = (1.0 - eta_d) * d_w + eta_d * obs_w;
    d_h = (1.0 - eta_d) * d_h + eta_d * obs_h;

    // Exact trust-weighted anchor update
    p_anchor_x = (1.0 - w_ema) * p_anchor_x + w_ema * obs_cx;
    p_anchor_y = (1.0 - w_ema) * p_anchor_y + w_ema * obs_cy;
    t_anchor = (1.0 - w_ema) * t_anchor + w_ema * t_sec;

    // Continuous health & liveness
    health = clamp_val((1.0 - alpha_h) * health + alpha_h * w_conf, 0.0, 1.0);
    conf = obs_conf;
    cls_id = obs_cls;
    k_lost = 0.0;
    ++hits;
    if (state == TrackState::Tentative && hits >= cfg.probation_frames) {
        state = TrackState::Confirmed;
    }
}

BBox PolarEMATrack::get_bbox() const {
    const double hw = 0.5 * d_w;
    const double hh = 0.5 * d_h;
    return BBox{pos_x - hw, pos_y - hh, pos_x + hw, pos_y + hh};
}

PolarEMATracker::PolarEMATracker(const Config& cfg) : cfg_(cfg) {
    cfg_.validate();
}

void PolarEMATracker::reset(double fps) {
    if (fps > 0.0 && std::isfinite(fps)) {
        cfg_.ref_fps = fps;
    }
    tracks_.clear();
    next_id_ = 1;
    frame_idx_ = 0;
    t_sec_ = 0.0;
}

void PolarEMATracker::set_image_size(int width, int height) {
    if (width > 0 && height > 0) {
        cfg_.img_width = width;
        cfg_.img_height = height;
    }
}

std::vector<TrackOutput> PolarEMATracker::update(
    const std::vector<BBox>& dets,
    const std::vector<double>& scores,
    const std::vector<int64_t>& cls_ids,
    double dt_sec) {
    ++frame_idx_;

    double dt_eff = 1.0 / cfg_.ref_fps;
    if (cfg_.dt_aware && dt_sec > 0.0 && std::isfinite(dt_sec)) {
        dt_eff = clamp_val(dt_sec, 1e-4, 30.0 / cfg_.ref_fps);
    }
    t_sec_ += dt_eff;

    const size_t n_dets = std::min(dets.size(), scores.size());
    std::vector<int64_t> clss(n_dets, 0);
    if (cls_ids.size() >= n_dets) {
        std::copy_n(cls_ids.begin(), n_dets, clss.begin());
    }

    std::vector<int> high_idx;
    std::vector<int> low_idx;
    high_idx.reserve(n_dets);
    low_idx.reserve(n_dets);
    for (size_t i = 0; i < n_dets; ++i) {
        const double sc = scores[i];
        if (sc >= cfg_.tau_high) {
            high_idx.push_back(static_cast<int>(i));
        } else if (sc >= cfg_.tau_low) {
            low_idx.push_back(static_cast<int>(i));
        }
    }

    // 1. Predict all existing tracks
    for (auto& trk : tracks_) {
        trk.predict(dt_eff, cfg_);
    }

    std::vector<PolarEMATrack*> confirmed;
    std::vector<PolarEMATrack*> tentative;
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Confirmed) {
            confirmed.push_back(&trk);
        } else {
            tentative.push_back(&trk);
        }
    }

    std::vector<bool> matched_confirmed(confirmed.size(), false);
    std::vector<bool> matched_tentative(tentative.size(), false);
    std::vector<int> track_assoc_det(tracks_.size(), -1);

    // Map confirmed/tentative pointers back to index in tracks_
    std::vector<size_t> conf_to_track_idx;
    std::vector<size_t> tent_to_track_idx;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].state == TrackState::Confirmed) {
            conf_to_track_idx.push_back(i);
        } else {
            tent_to_track_idx.push_back(i);
        }
    }

    auto compute_max_other_iou = [&](const BBox& det_box, int self_conf_idx) -> double {
        double max_ov = 0.0;
        for (int k = 0; k < static_cast<int>(confirmed.size()); ++k) {
            if (k == self_conf_idx) continue;
            max_ov = std::max(max_ov, compute_iou(det_box, confirmed[static_cast<size_t>(k)]->get_bbox()));
        }
        return max_ov;
    };

    std::vector<int> u_conf_idx;
    for (int i = 0; i < static_cast<int>(confirmed.size()); ++i) u_conf_idx.push_back(i);
    std::vector<int> rem_high_idx = high_idx;

    // 2. Stage 1: Confirmed + lost tracks <-> High-confidence detections
    if (!confirmed.empty() && !rem_high_idx.empty()) {
        const auto cost1 = iou_polar_cost_matrix(
            confirmed, dets, clss, rem_high_idx, t_sec_, dt_eff, cfg_, cfg_.iou_thresh_high);
        const auto res1 = linear_assignment(
            cost1, static_cast<int>(confirmed.size()), static_cast<int>(rem_high_idx.size()), 1.0 + cfg_.iou_thresh_high);

        // Precompute overlap against all predicted confirmed boxes before mutating positions
        std::vector<double> ov_list;
        ov_list.reserve(res1.matches.size());
        for (const auto& m : res1.matches) {
            const int g_di = rem_high_idx[static_cast<size_t>(m.second)];
            ov_list.push_back(compute_max_other_iou(dets[static_cast<size_t>(g_di)], m.first));
        }
        for (size_t k = 0; k < res1.matches.size(); ++k) {
            const int ti = res1.matches[k].first;
            const int g_di = rem_high_idx[static_cast<size_t>(res1.matches[k].second)];
            confirmed[static_cast<size_t>(ti)]->update(
                dets[static_cast<size_t>(g_di)],
                scores[static_cast<size_t>(g_di)],
                clss[static_cast<size_t>(g_di)],
                t_sec_,
                dt_eff,
                ov_list[k],
                cfg_);
            matched_confirmed[static_cast<size_t>(ti)] = true;
            track_assoc_det[conf_to_track_idx[static_cast<size_t>(ti)]] = g_di;
        }
        u_conf_idx = res1.unmatched_rows;
        std::vector<int> next_high;
        next_high.reserve(res1.unmatched_cols.size());
        for (int c : res1.unmatched_cols) next_high.push_back(rem_high_idx[static_cast<size_t>(c)]);
        rem_high_idx = std::move(next_high);
    }

    // 3. Stage 2: Remaining active (k_lost == 0) confirmed tracks <-> Low-confidence detections
    std::vector<int> rem_active_idx;
    std::vector<PolarEMATrack*> rem_active;
    for (int ti : u_conf_idx) {
        if (confirmed[static_cast<size_t>(ti)]->k_lost == 0.0) {
            rem_active_idx.push_back(ti);
            rem_active.push_back(confirmed[static_cast<size_t>(ti)]);
        }
    }
    if (!rem_active.empty() && !low_idx.empty()) {
        const auto cost2 = iou_polar_cost_matrix(
            rem_active, dets, clss, low_idx, t_sec_, dt_eff, cfg_, cfg_.iou_thresh_low);
        const auto res2 = linear_assignment(
            cost2, static_cast<int>(rem_active.size()), static_cast<int>(low_idx.size()), 1.0 + cfg_.iou_thresh_high);

        std::vector<double> ov_list;
        ov_list.reserve(res2.matches.size());
        for (const auto& m : res2.matches) {
            const int ti = rem_active_idx[static_cast<size_t>(m.first)];
            const int g_di = low_idx[static_cast<size_t>(m.second)];
            ov_list.push_back(compute_max_other_iou(dets[static_cast<size_t>(g_di)], ti));
        }
        for (size_t k = 0; k < res2.matches.size(); ++k) {
            const int ti = rem_active_idx[static_cast<size_t>(res2.matches[k].first)];
            const int g_di = low_idx[static_cast<size_t>(res2.matches[k].second)];
            confirmed[static_cast<size_t>(ti)]->update(
                dets[static_cast<size_t>(g_di)],
                scores[static_cast<size_t>(g_di)],
                clss[static_cast<size_t>(g_di)],
                t_sec_,
                dt_eff,
                ov_list[k],
                cfg_);
            matched_confirmed[static_cast<size_t>(ti)] = true;
            track_assoc_det[conf_to_track_idx[static_cast<size_t>(ti)]] = g_di;
        }
    }

    // 4. Stage 3: Remaining lost (k_lost > 0) confirmed tracks <-> Remaining high-conf detections
    std::vector<int> rem_lost_idx;
    std::vector<PolarEMATrack*> rem_lost;
    for (int ti : u_conf_idx) {
        if (!matched_confirmed[static_cast<size_t>(ti)] && confirmed[static_cast<size_t>(ti)]->k_lost > 0.0) {
            rem_lost_idx.push_back(ti);
            rem_lost.push_back(confirmed[static_cast<size_t>(ti)]);
        }
    }
    if (!rem_lost.empty() && !rem_high_idx.empty()) {
        const auto cost3 = relocation_polar_cost_matrix(
            rem_lost, dets, clss, rem_high_idx, t_sec_, dt_eff, cfg_);
        const auto res3 = linear_assignment(
            cost3, static_cast<int>(rem_lost.size()), static_cast<int>(rem_high_idx.size()), 1.0);

        std::vector<double> ov_list;
        ov_list.reserve(res3.matches.size());
        for (const auto& m : res3.matches) {
            const int ti = rem_lost_idx[static_cast<size_t>(m.first)];
            const int g_di = rem_high_idx[static_cast<size_t>(m.second)];
            ov_list.push_back(compute_max_other_iou(dets[static_cast<size_t>(g_di)], ti));
        }
        for (size_t k = 0; k < res3.matches.size(); ++k) {
            const int ti = rem_lost_idx[static_cast<size_t>(res3.matches[k].first)];
            const int g_di = rem_high_idx[static_cast<size_t>(res3.matches[k].second)];
            confirmed[static_cast<size_t>(ti)]->update(
                dets[static_cast<size_t>(g_di)],
                scores[static_cast<size_t>(g_di)],
                clss[static_cast<size_t>(g_di)],
                t_sec_,
                dt_eff,
                ov_list[k],
                cfg_);
            matched_confirmed[static_cast<size_t>(ti)] = true;
            track_assoc_det[conf_to_track_idx[static_cast<size_t>(ti)]] = g_di;
        }
        std::vector<int> next_high;
        next_high.reserve(res3.unmatched_cols.size());
        for (int c : res3.unmatched_cols) next_high.push_back(rem_high_idx[static_cast<size_t>(c)]);
        rem_high_idx = std::move(next_high);
    }

    // 5. Stage 4: Tentative tracks <-> Remaining high-conf detections (pure IoU)
    if (!tentative.empty() && !rem_high_idx.empty()) {
        const int n_tent = static_cast<int>(tentative.size());
        const int m_rem = static_cast<int>(rem_high_idx.size());
        std::vector<double> cost4(static_cast<size_t>(n_tent) * static_cast<size_t>(m_rem), 0.0);
        for (int i = 0; i < n_tent; ++i) {
            const PolarEMATrack& trk = *tentative[static_cast<size_t>(i)];
            const BBox tb = trk.get_bbox();
            for (int j = 0; j < m_rem; ++j) {
                const int g_di = rem_high_idx[static_cast<size_t>(j)];
                const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(m_rem) + static_cast<size_t>(j);
                if (cfg_.check_class_match && trk.cls_id != clss[static_cast<size_t>(g_di)]) {
                    cost4[idx] = std::numeric_limits<double>::infinity();
                } else {
                    cost4[idx] = 1.0 - compute_iou(tb, dets[static_cast<size_t>(g_di)]);
                }
            }
        }
        const auto res4 = linear_assignment(cost4, n_tent, m_rem, 1.0 - cfg_.iou_thresh_high);
        for (const auto& m : res4.matches) {
            const int ti = m.first;
            const int g_di = rem_high_idx[static_cast<size_t>(m.second)];
            tentative[static_cast<size_t>(ti)]->update(
                dets[static_cast<size_t>(g_di)],
                scores[static_cast<size_t>(g_di)],
                clss[static_cast<size_t>(g_di)],
                t_sec_,
                dt_eff,
                0.0,
                cfg_);
            matched_tentative[static_cast<size_t>(ti)] = true;
            track_assoc_det[tent_to_track_idx[static_cast<size_t>(ti)]] = g_di;
        }
        std::vector<int> next_high;
        next_high.reserve(res4.unmatched_cols.size());
        for (int c : res4.unmatched_cols) next_high.push_back(rem_high_idx[static_cast<size_t>(c)]);
        rem_high_idx = std::move(next_high);
    }

    // 6. Mark unmatched tracks missed & prune expired / border-exited tracks
    for (size_t i = 0; i < confirmed.size(); ++i) {
        if (!matched_confirmed[i]) confirmed[i]->mark_missed(dt_eff, cfg_);
    }
    for (size_t i = 0; i < tentative.size(); ++i) {
        if (!matched_tentative[i]) tentative[i]->mark_missed(dt_eff, cfg_);
    }

    const double img_w = static_cast<double>(cfg_.img_width);
    const double img_h = static_cast<double>(cfg_.img_height);

    std::vector<PolarEMATrack> kept_tracks;
    std::vector<int> kept_assoc;
    kept_tracks.reserve(tracks_.size() + rem_high_idx.size());
    kept_assoc.reserve(tracks_.size() + rem_high_idx.size());

    for (size_t i = 0; i < tracks_.size(); ++i) {
        const PolarEMATrack& trk = tracks_[i];
        if (trk.state == TrackState::Tentative && trk.k_lost > static_cast<double>(cfg_.early_term_lost)) {
            continue;
        }
        if (trk.k_lost > static_cast<double>(cfg_.max_lost_frames)) {
            continue;
        }
        if (trk.state == TrackState::Confirmed && trk.k_lost > 0.0 &&
            cfg_.border_exit_lost > 0 && img_w > 0.0 && img_h > 0.0) {
            const BBox pbox = trk.get_bbox();
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
        kept_tracks.push_back(trk);
        kept_assoc.push_back(track_assoc_det[i]);
    }
    tracks_ = std::move(kept_tracks);

    // 7. Spawn new tentative tracks from remaining high-conf detections (live suppression)
    const double spawn_thresh = cfg_.effective_new_track_thresh();
    for (int g_di : rem_high_idx) {
        const BBox& det_box = dets[static_cast<size_t>(g_di)];
        const double det_conf = scores[static_cast<size_t>(g_di)];
        if (det_conf < spawn_thresh) continue;
        const int64_t det_cls = clss[static_cast<size_t>(g_di)];

        bool suppress = false;
        for (const auto& existing : tracks_) {
            if (existing.k_lost > 0.0) continue;
            if (cfg_.check_class_match && existing.cls_id != det_cls) continue;
            if (compute_iou(det_box, existing.get_bbox()) > cfg_.min_iou_diff_new) {
                suppress = true;
                break;
            }
        }
        if (suppress) continue;

        PolarEMATrack trk(next_id_++, det_box, det_conf, det_cls, t_sec_, cfg_);
        if (frame_idx_ == 1 || cfg_.probation_frames == 1) {
            trk.state = TrackState::Confirmed;
            trk.hits = std::max(trk.hits, cfg_.probation_frames);
        }
        tracks_.push_back(std::move(trk));
        kept_assoc.push_back(g_di);
    }

    // 8. Emit active tracks BEFORE shadow tracks so downstream capacity limits never drop active targets
    std::vector<TrackOutput> outputs;
    outputs.reserve(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) {
        const PolarEMATrack& trk = tracks_[i];
        if (trk.state != TrackState::Confirmed || trk.k_lost != 0.0) continue;
        outputs.push_back(TrackOutput{
            trk.id, trk.get_bbox(), trk.conf, trk.cls_id, false, trk.age, trk.k_lost, kept_assoc[i]});
    }
    if (cfg_.emit_shadow_tracks) {
        for (size_t i = 0; i < tracks_.size(); ++i) {
            const PolarEMATrack& trk = tracks_[i];
            if (trk.state != TrackState::Confirmed || trk.k_lost <= 0.0) continue;
            if (cfg_.max_shadow_emit_lost > 0 &&
                trk.k_lost > static_cast<double>(cfg_.max_shadow_emit_lost)) {
                continue;
            }
            const BBox out_box = trk.get_bbox();
            if (cfg_.border_exit_lost > 0 && trk.k_lost >= 1.0 && img_w > 0.0 && img_h > 0.0) {
                const double m = cfg_.border_exit_margin_px;
                const bool at_border =
                    (out_box.x1 <= m) || (out_box.y1 <= m) ||
                    (out_box.x2 >= img_w - m) || (out_box.y2 >= img_h - m);
                if (at_border) continue;
            }
            outputs.push_back(TrackOutput{
                trk.id, out_box, trk.conf, trk.cls_id, true, trk.age, trk.k_lost, -1});
        }
    }
    return outputs;
}

}  // namespace vema2
