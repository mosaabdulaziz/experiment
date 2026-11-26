#include "vema2_tracker.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace vema2;

static BBox box_cxcywh(double cx, double cy, double w, double h) {
    return BBox{cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h};
}

static void test_hungarian_premask() {
    std::vector<double> cost = {0.10, 0.74, 0.80, 10.0};
    auto res = linear_assignment(cost, 2, 2, 0.75);
    assert(res.matches.size() == 1);
    assert(res.matches[0].first == 0 && res.matches[0].second == 0);
    assert(res.unmatched_rows.size() == 1 && res.unmatched_rows[0] == 1);
    assert(res.unmatched_cols.size() == 1 && res.unmatched_cols[0] == 1);
}

static void test_anchor_invariant() {
    Config cfg;
    cfg.ref_fps = 30.0;
    cfg.dt_aware = true;
    const double dt = 1.0 / 30.0;
    const double vx = 300.0;
    const double vy = -120.0;
    const double x0 = 200.0;
    const double y0 = 400.0;

    PolarEMATrack trk(1, box_cxcywh(x0, y0, 60.0, 150.0), 0.9, 0, 0.0, cfg);
    double t_sec = 0.0;
    for (int k = 1; k < 20; ++k) {
        t_sec += dt;
        trk.predict(dt, cfg);
        const double px = x0 + vx * t_sec;
        const double py = y0 + vy * t_sec;
        const double dt_a = t_sec - trk.t_anchor;
        const double v_obs_x = (px - trk.p_anchor_x) / dt_a;
        const double v_obs_y = (py - trk.p_anchor_y) / dt_a;
        assert(std::abs(v_obs_x - vx) < 1e-8);
        assert(std::abs(v_obs_y - vy) < 1e-8);
        const double conf = 0.20 + 0.04 * (k % 15);
        const double ov = 0.05 * (k % 10);
        trk.update(box_cxcywh(px, py, 60.0, 150.0), conf, 0, t_sec, dt, ov, cfg);
    }
}

static void test_high_conf_crossing_swap() {
    Config cfg;
    PolarEMATracker tracker(cfg);
    const double dt = 1.0 / 30.0;
    std::set<uint64_t> ids;
    uint64_t id_left_start = 0, id_right_start = 0;
    uint64_t id_left_end = 0, id_right_end = 0;

    for (int f = 1; f <= 70; ++f) {
        const double t = static_cast<double>(f - 1);
        std::vector<BBox> dets;
        std::vector<double> scores;
        std::vector<int64_t> clss;

        // Ped 1: 400 -> 1090 (+10/frame)
        double cx1 = 400.0 + 10.0 * t;
        if (f >= 29 && f <= 33) cx1 -= 6.0;
        dets.push_back(box_cxcywh(cx1, 500.0, 60.0, 150.0));
        scores.push_back(0.86);
        clss.push_back(0);

        // Ped 2: 1000 -> 310 (-10/frame), occluded frames 30..33
        if (!(f >= 30 && f <= 33)) {
            dets.push_back(box_cxcywh(1000.0 - 10.0 * t, 500.0, 60.0, 150.0));
            scores.push_back(0.85);
            clss.push_back(0);
        }

        auto outs = tracker.update(dets, scores, clss, dt);
        for (const auto& o : outs) {
            ids.insert(o.track_id);
        }
        if (f == 5) {
            for (const auto& o : outs) {
                double cx = 0.5 * (o.bbox.x1 + o.bbox.x2);
                if (cx < 700.0) id_left_start = o.track_id;
                else id_right_start = o.track_id;
            }
        }
        if (f == 65) {
            for (const auto& o : outs) {
                double cx = 0.5 * (o.bbox.x1 + o.bbox.x2);
                if (cx > 700.0) id_left_end = o.track_id;
                else id_right_end = o.track_id;
            }
        }
    }
    assert(ids.size() == 2);
    assert(id_left_start != 0 && id_left_start == id_left_end);
    assert(id_right_start != 0 && id_right_start == id_right_end);
}

static void test_overtaking_same_direction() {
    Config cfg;
    PolarEMATracker tracker(cfg);
    const double dt = 1.0 / 30.0;
    std::set<uint64_t> ids;
    uint64_t slow_start = 0, fast_start = 0, slow_end = 0, fast_end = 0;

    for (int f = 1; f <= 80; ++f) {
        const double t = static_cast<double>(f - 1);
        std::vector<BBox> dets;
        std::vector<double> scores;
        std::vector<int64_t> clss;

        if (!(f >= 33 && f <= 36)) {
            dets.push_back(box_cxcywh(450.0 + 2.5 * t, 500.0, 58.0, 148.0));
            scores.push_back(0.87);
            clss.push_back(0);
        }
        dets.push_back(box_cxcywh(200.0 + 10.0 * t, 500.0, 58.0, 148.0));
        scores.push_back(0.88);
        clss.push_back(0);

        auto outs = tracker.update(dets, scores, clss, dt);
        for (const auto& o : outs) ids.insert(o.track_id);
        if (f == 5) {
            for (const auto& o : outs) {
                double cx = 0.5 * (o.bbox.x1 + o.bbox.x2);
                if (cx > 350.0) slow_start = o.track_id;
                else fast_start = o.track_id;
            }
        }
        if (f == 75) {
            for (const auto& o : outs) {
                double cx = 0.5 * (o.bbox.x1 + o.bbox.x2);
                if (cx < 750.0) slow_end = o.track_id;
                else fast_end = o.track_id;
            }
        }
    }
    assert(ids.size() == 2);
    assert(slow_start != 0 && slow_start == slow_end);
    assert(fast_start != 0 && fast_start == fast_end);
}

static void test_standing_to_seated() {
    Config cfg;
    PolarEMATracker tracker(cfg);
    const double dt = 1.0 / 30.0;
    std::set<uint64_t> ids;
    double final_h = 0.0;

    for (int f = 1; f <= 110; ++f) {
        BBox b = (f <= 30) ? box_cxcywh(600.0, 500.0, 70.0, 170.0)
                           : box_cxcywh(600.0, 545.0, 75.0, 80.0);
        auto outs = tracker.update({b}, {0.88}, {0}, dt);
        for (const auto& o : outs) {
            ids.insert(o.track_id);
            final_h = o.bbox.y2 - o.bbox.y1;
        }
    }
    assert(ids.size() == 1);
    assert(final_h < 105.0);
}

static void test_yaml_shadow_and_border_exit() {
    Config cfg = Config::from_yaml_file("perception/pipeline/config_tracker_vema2.yml");
    assert(std::abs(cfg.tau_high - 0.60) < 1e-6);
    assert(std::abs(cfg.tau_mem - 0.15) < 1e-6);
    assert(cfg.max_lost_frames == 600);
    assert(cfg.emit_shadow_tracks == true);
    assert(cfg.max_shadow_emit_lost == 1);
    assert(cfg.max_targets_per_stream == 120);

    // Ensure invalid tau_low == tau_high does not produce NaN
    Config bad = Config::from_yaml_string("tau_low: 0.6\ntau_high: 0.6\n");
    assert(bad.tau_low < bad.tau_high);

    PolarEMATracker tracker(cfg);
    tracker.set_image_size(1080, 720);
    const double dt = 1.0 / 30.0;

    // Track exits right border (x ~ 1060): border shadow suppression keeps shadow_count == 0
    for (int f = 1; f <= 5; ++f) {
        auto outs = tracker.update({box_cxcywh(980.0 + 15.0 * f, 360.0, 60.0, 150.0)}, {0.9}, {0}, dt);
        assert(outs.size() == 1 && !outs[0].is_shadow);
    }
    int border_shadow_count = 0;
    for (int f = 6; f <= 15; ++f) {
        auto outs = tracker.update({}, {}, {}, dt);
        for (const auto& o : outs) {
            if (o.is_shadow) ++border_shadow_count;
            assert(std::isfinite(o.bbox.x1));
        }
    }
    assert(border_shadow_count == 0);

    // Mid-frame lost track still emits shadow boxes up to max_shadow_emit_lost
    tracker.reset(30.0);
    for (int f = 1; f <= 4; ++f) {
        auto outs = tracker.update({box_cxcywh(500.0 + 4.0 * f, 360.0, 60.0, 150.0)}, {0.9}, {0}, dt);
        assert(outs.size() == 1 && !outs[0].is_shadow);
    }
    int mid_shadow_count = 0;
    for (int f = 5; f <= 12; ++f) {
        auto outs = tracker.update({}, {}, {}, dt);
        for (const auto& o : outs) {
            if (o.is_shadow) ++mid_shadow_count;
        }
    }
    assert(mid_shadow_count == cfg.max_shadow_emit_lost);
}

static void test_python_parity_trace() {
    Config cfg;
    PolarEMATracker tracker(cfg);
    const double dt = 1.0 / 30.0;
    std::vector<TrackOutput> last_outs;
    for (int f = 1; f <= 15; ++f) {
        double t = static_cast<double>(f - 1);
        std::vector<BBox> dets = {
            box_cxcywh(200.0 + 8.0 * t, 300.0 + 2.0 * t, 60.0, 150.0),
            box_cxcywh(800.0 - 6.0 * t, 300.0 + 1.0 * t, 62.0, 148.0),
        };
        last_outs = tracker.update(dets, {0.9, 0.85}, {0, 0}, dt);
    }
    assert(last_outs.size() == 2);
    std::printf("PARITY_F15: ");
    for (const auto& o : last_outs) {
        std::printf("[id=%llu x1=%.6f y1=%.6f x2=%.6f y2=%.6f] ",
                    static_cast<unsigned long long>(o.track_id),
                    o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2);
    }
    std::printf("\n");
}

int main() {
    test_hungarian_premask();
    test_anchor_invariant();
    test_high_conf_crossing_swap();
    test_overtaking_same_direction();
    test_standing_to_seated();
    test_yaml_shadow_and_border_exit();
    test_python_parity_trace();
    std::printf("ALL C++ VEMA2 TESTS PASSED\n");
    return 0;
}
