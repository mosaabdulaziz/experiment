import pytest

from policy import DEFAULT_POLICY, effective, validate_policy, validate_policy_update


def test_effective_overlays_known_numbers_only():
    p = effective({"seat_dwell_s": 12, "unknown": 1})
    assert p["seat_dwell_s"] == 12.0 and "unknown" not in p
    assert p["chair_gap_s"] == DEFAULT_POLICY["chair_gap_s"]


def test_new_knobs_exist():
    for k in ("sync_window_s", "reid_match_min", "reid_veto", "reid_margin", "reattach_s", "staff_threshold",
              "staff_min_samples", "still_window_s", "still_radius", "seat_dwell_s", "seat_jitter_s",
              "wash_return_s"):
        assert k in DEFAULT_POLICY


def test_bounded_knobs_are_rejected_above_their_maximum():
    with pytest.raises(ValueError, match="<= 1"):
        validate_policy_update({"staff_threshold": 1.5})
    for k in ("still_box_frac", "seat_staff_threshold"):
        with pytest.raises(ValueError, match="<= 1"):
            validate_policy_update({k: 1.5})


def test_knobs_with_a_minimum_are_rejected_below_it():
    # 2026-09-26 (final B review I4): staff_min_samples 0 divided by zero every cycle
    with pytest.raises(ValueError, match=">= 1"):
        validate_policy_update({"staff_min_samples": 0})
    assert validate_policy_update({"staff_min_samples": 1}) == {"staff_min_samples": 1.0}


def test_camera_dead_s_exists():
    assert DEFAULT_POLICY["camera_dead_s"] == 150.0


def test_staff_threshold_default_is_the_classifier_cut():
    """Project A (2026-09-26): the trained classifier's calibrated staff_p; at
    0.70 on val, 92.0% of staff crops and 1.1% of customer crops pass."""
    from policy import DEFAULT_POLICY, MAXIMUMS
    assert DEFAULT_POLICY["staff_threshold"] == 0.70
    assert DEFAULT_POLICY["staff_threshold"] <= MAXIMUMS["staff_threshold"]
    assert DEFAULT_POLICY["seat_staff_threshold"] == 0.9


def test_identity_switches_default_to_todays_behaviour():
    # 2026-09-27 (39-min testbed spec, section 4): 0 = today's behaviour for all three
    from policy import MAXIMUMS
    for k, top in (("create_in_owned_areas", 1.0), ("identity_model", 2.0), ("reid_gallery", 2.0)):
        assert DEFAULT_POLICY[k] == 0.0 and MAXIMUMS[k] == top


def test_identity_switches_are_whole_numbers_within_their_range():
    with pytest.raises(ValueError, match="<= 2"):
        validate_policy_update({"identity_model": 3})
    for k, v in (("reid_gallery", 1.5), ("create_in_owned_areas", 0.5), ("identity_model", "1.2")):
        with pytest.raises(ValueError, match="whole number"):
            validate_policy_update({k: v})
    assert validate_policy_update({"reid_gallery": 2, "identity_model": 1.0}) == \
        {"reid_gallery": 2.0, "identity_model": 1.0}


def test_chair_resume_is_off_by_default_with_its_similarity_bounds():
    # 2026-09-28 (39-min testbed spec, 6.1)
    from policy import MAXIMUMS
    assert DEFAULT_POLICY["chair_resume_s"] == 0.0
    assert (DEFAULT_POLICY["resume_sim_same"], DEFAULT_POLICY["resume_sim_diff"]) == (0.70, 0.50)
    for k in ("resume_sim_same", "resume_sim_diff"):
        assert MAXIMUMS[k] == 1.0
        with pytest.raises(ValueError, match="<= 1"):
            validate_policy_update({k: 1.5})


def test_staff_release_mode_is_off_by_default_and_a_whole_number_up_to_1():
    # 2026-09-28 (39-min testbed spec, 6.2)
    from policy import MAXIMUMS
    assert DEFAULT_POLICY["staff_release_mode"] == 0.0 and MAXIMUMS["staff_release_mode"] == 1.0
    with pytest.raises(ValueError, match="<= 1"):
        validate_policy_update({"staff_release_mode": 2})
    with pytest.raises(ValueError, match="whole number"):
        validate_policy_update({"staff_release_mode": 0.5})


def test_handoff_rule_is_off_by_default_and_a_whole_number_up_to_4():
    # 2026-09-28 (39-min testbed spec, 6.3)
    from policy import MAXIMUMS
    assert DEFAULT_POLICY["handoff_rule"] == 0.0 and MAXIMUMS["handoff_rule"] == 4.0
    with pytest.raises(ValueError, match="<= 4"):
        validate_policy_update({"handoff_rule": 5})
    with pytest.raises(ValueError, match="whole number"):
        validate_policy_update({"handoff_rule": 0.5})


def test_staff_window_and_latched_release_default_to_today():
    # 2026-09-29: faster staff correction, off by default
    from policy import MAXIMUMS, MINIMUMS
    assert DEFAULT_POLICY["staff_window"] == 15.0 and DEFAULT_POLICY["staff_latched_release"] == 0.0
    assert MAXIMUMS["staff_window"] == 15.0 and MAXIMUMS["staff_latched_release"] == 15.0
    assert MINIMUMS["staff_window"] == 1.0
    with pytest.raises(ValueError, match="<= 15"):
        validate_policy_update({"staff_window": 16})
    with pytest.raises(ValueError, match="whole number"):
        validate_policy_update({"staff_latched_release": 2.5})


def test_chair_refresh_run_defaults_and_bounds():
    # 2026-10-04 (Task 16c)
    assert DEFAULT_POLICY["ce_refresh_run_n"] == 3.0 and DEFAULT_POLICY["ce_refresh_after_s"] == 30.0
    for bad in (0, 2.5):
        with pytest.raises(ValueError):
            validate_policy_update({"ce_refresh_run_n": bad})
    assert validate_policy_update({"ce_refresh_run_n": 1}) == {"ce_refresh_run_n": 1.0}


def test_chair_engine_knobs_where_0_breaks_the_engine_have_a_floor():
    """2026-10-04 (final review I-2): ce_start_window_s 0 crashed every cycle; dropout/end gap 0 silently
    stop sittings; seat/carry shares 0 make anyone anywhere occupy or cover a chair."""
    for k in ("ce_start_window_s", "ce_dropout_s", "ce_end_gap_s", "ce_seat_min", "ce_carry_outline", "ce_carry_seat"):
        with pytest.raises(ValueError):
            validate_policy_update({k: 0})
        assert validate_policy_update({k: DEFAULT_POLICY[k]}) == {k: DEFAULT_POLICY[k]}
    for k in ("ce_resume_window_s", "ce_min_haircut_s", "ce_door_confirm_s", "ce_door_lost_s", "ce_carry_max_s"):
        assert validate_policy_update({k: 0}) == {k: 0.0}      # 0 is a meaningful setting there


def test_start_chain_and_inside_gate_knobs_allow_0_as_off():
    """2026-10-05 (haircut accuracy): both fixes have an off value (0) and the gate is a share."""
    assert "ce_start_chain_s" in DEFAULT_POLICY and "ce_seat_inside_min" in DEFAULT_POLICY
    assert validate_policy_update({"ce_start_chain_s": 0, "ce_seat_inside_min": 0}) == {
        "ce_start_chain_s": 0.0, "ce_seat_inside_min": 0.0}
    with pytest.raises(ValueError):
        validate_policy_update({"ce_seat_inside_min": 1.01})
    with pytest.raises(ValueError):
        validate_policy_update({"ce_start_chain_s": -1})


def test_start_chain_bounds():
    """Review fixes 1-2: the chain gap is capped; the chain age cap has a floor (0 would be a dead chain)."""
    assert DEFAULT_POLICY["ce_start_chain_max_s"] == 240.0
    for bad in ({"ce_start_chain_s": 601}, {"ce_start_chain_max_s": 0}, {"ce_start_chain_max_s": 3601}):
        with pytest.raises(ValueError):
            validate_policy_update(bad)
    assert validate_policy_update({"ce_start_chain_s": 600}) == {"ce_start_chain_s": 600.0}


# 2026-10-05 (far-chair per-chair rules): "<base key>@<chair>" overrides one chair-engine knob for one chair.
def test_per_chair_override_keys_are_accepted_for_per_chair_knobs_only():
    from policy import PER_CHAIR_KEYS
    assert PER_CHAIR_KEYS == {"ce_seat_min", "ce_conf_min", "ce_seat_inside_min", "ce_start_window_s",
                              "ce_start_frac", "ce_end_gap_s",
                              # 2026-10-18 (user: L5, L6, R1, L7 go away while their barber hides the customer)
                              "ce_carry_max_s", "ce_carry_lower_frac", "ce_carry_outline", "ce_carry_seat",
                              "ce_cover_min_frac",       # 2026-11-13 (user: cover flickers held L7's empty seat)
                              "ce_low_occupancy_short",
                              "ce_low_occupancy",        # 2026-11-14 (user: a haircut seen on the seat in few frames)
                              # 2026-10-18 (user: re-ID thresholds per chair; the identity check after cover)
                              "ce_resume_sim_low", "ce_resume_sim_high", "ce_reseat_check_s",
                              # 2026-10-18 (user: a bag on VIP2 was a haircut)
                              "ce_object_spread",
                              # 2026-10-22 (user: clothes on VIP2 were a haircut)
                              "ce_object_conf",
                              # 2026-11-10 (user: staff members' own haircuts, on the VIP chairs)
                              "ce_staff_cut_check", "ce_staff_cut_share"}
    assert validate_policy_update({"ce_seat_min@R9": 0.08, "ce_end_gap_s@R9": "400"}) == {
        "ce_seat_min@R9": 0.08, "ce_end_gap_s@R9": 400.0}
    for bad in ("ce_lower_frac@R9", "bogus@R9", "ce_seat_min@", "@R9", "ce_seat_min@R9@x", "ce_seat_min@ R9"):
        with pytest.raises(ValueError, match="unknown policy key"):
            validate_policy_update({bad: 0.1})


def test_per_chair_overrides_have_their_base_keys_bounds():
    for k, v in (("ce_seat_min@R9", 0), ("ce_seat_min@R9", 1.5), ("ce_end_gap_s@R9", 0), ("ce_start_frac@R9", 2),
                 ("ce_start_window_s@R9", 0.5), ("ce_conf_min@R9", -0.1), ("ce_seat_inside_min@R9", 1.01)):
        with pytest.raises(ValueError, match=r"[<>]="):
            validate_policy_update({k: v})
    with pytest.raises(ValueError, match="number"):
        validate_policy_update({"ce_seat_min@R9": "x"})


def test_effective_keeps_valid_overrides_and_drops_unknown_ones():
    p = effective({"ce_seat_min@R9": 0.07, "ce_lower_frac@R9": 0.5, "bogus@R9": 1})
    assert p["ce_seat_min@R9"] == 0.07
    assert "ce_lower_frac@R9" not in p and "bogus@R9" not in p


def test_policy_for_applies_a_chairs_overrides_only_to_that_chair():
    from policy import policy_for
    base = effective({"ce_seat_min@R9": 0.07, "ce_end_gap_s@R9": 300, "ce_seat_min@R8": 0.2})
    p9, p1 = policy_for(base, "R9"), policy_for(base, "R1")
    assert p9["ce_seat_min"] == 0.07 and p9["ce_end_gap_s"] == 300.0
    assert p1["ce_seat_min"] == base["ce_seat_min"] and p1["ce_end_gap_s"] == base["ce_end_gap_s"]
    assert base["ce_seat_min"] == DEFAULT_POLICY["ce_seat_min"]          # not mutated


def test_overrides_hot_reload_from_the_stored_policy(tmp_path):
    from policy import policy_for, policy_store
    store = policy_store(tmp_path)
    store.save({**effective(store.load()), "ce_end_gap_s@R9": 250.0})
    assert policy_for(effective(store.load()), "R9")["ce_end_gap_s"] == 250.0
    store.save({**effective(store.load()), "ce_end_gap_s@R9": 120.0})
    assert policy_for(effective(store.load()), "R9")["ce_end_gap_s"] == 120.0


def test_shipped_l5_overrides_and_other_chairs_unchanged():
    """2026-10-05 (far-chair sweep): L5 gets its own seat share, start window and end gap; 2026-10-12: L4 its start
    window (L5's); no other chair is overridden."""
    from policy import policy_for
    l5 = policy_for(DEFAULT_POLICY, "L5")
    assert (l5["ce_seat_min"], l5["ce_start_window_s"], l5["ce_start_frac"], l5["ce_end_gap_s"]) == (0.08, 30.0, 0.30, 400.0)
    l4 = policy_for(DEFAULT_POLICY, "L4")
    assert (l4["ce_seat_min"], l4["ce_start_window_s"], l4["ce_start_frac"], l4["ce_end_gap_s"]) == (0.10, 30.0, 0.30, 60.0)
    for chair in ("L1", "L2", "L3", "R1", "R2", "R3", "R4", "R5"):
        assert policy_for(DEFAULT_POLICY, chair) is DEFAULT_POLICY
    assert (DEFAULT_POLICY["ce_seat_min"], DEFAULT_POLICY["ce_end_gap_s"]) == (0.10, 60.0)
    over = {k: v for k, v in DEFAULT_POLICY.items() if "@" in k}
    assert validate_policy_update(over) == over


def test_wait_settings_have_defaults_and_bounds():
    assert (DEFAULT_POLICY["wt_start_s"], DEFAULT_POLICY["wt_row_min"], DEFAULT_POLICY["wt_row_s"],
            DEFAULT_POLICY["wt_lost_s"], DEFAULT_POLICY["wt_reconnect_s"],
            DEFAULT_POLICY["wt_reconnect_dist"]) == (9.0, 0.5, 3.0, 2.0, 10.0, 0.10)
    for key in ("wt_start_s", "wt_lost_s", "wt_reconnect_dist", "wt_row_min"):
        with pytest.raises(ValueError):
            validate_policy_update({key: 0})
    for key in ("wt_row_min", "wt_reconnect_dist"):
        with pytest.raises(ValueError):
            validate_policy_update({key: 1.5})
    assert validate_policy_update({"wt_row_s": 0, "wt_reconnect_s": 0}) == {"wt_row_s": 0, "wt_reconnect_s": 0}


def test_wait_start_confidence_default_and_bounds():
    assert DEFAULT_POLICY["wt_conf_min"] == 0.6
    with pytest.raises(ValueError):
        validate_policy_update({"wt_conf_min": 1.5})
    assert validate_policy_update({"wt_conf_min": 0}) == {"wt_conf_min": 0}


def test_zone_display_switches_default_on_and_are_whole_numbers():
    keys = ("zone_show_entrance_line", "zone_show_chair", "zone_show_seat", "zone_show_waiting",
            "zone_show_washing", "zone_show_row")
    assert all(DEFAULT_POLICY[k] == 1.0 for k in keys)
    assert validate_policy_update({"zone_show_row": 0})["zone_show_row"] == 0
    for bad in (0.5, 2):
        with pytest.raises(ValueError):
            validate_policy_update({"zone_show_row": bad})


def test_face_register_knobs():
    assert (DEFAULT_POLICY["fr_before_s"], DEFAULT_POLICY["fr_after_s"], DEFAULT_POLICY["fr_keep_s"],
            DEFAULT_POLICY["zone_show_door_area"]) == (8, 30, 20, 1)
    assert validate_policy_update({"fr_after_s": 0}) == {"fr_after_s": 0.0}
    with pytest.raises(ValueError, match=">= 1"):
        validate_policy_update({"fr_keep_s": 0})
    with pytest.raises(ValueError):
        validate_policy_update({"zone_show_door_area": 0.5})


def test_register_source_defaults_to_door_faces_and_is_a_zero_or_one_switch():
    assert DEFAULT_POLICY["register_source"] == 1.0
    assert validate_policy_update({"register_source": 1})["register_source"] == 1
    for bad in (0.5, 3):
        with pytest.raises(ValueError):
            validate_policy_update({"register_source": bad})


from policy import policy_store


def test_effective_overlays_stored_on_a_site_base():
    base = {**DEFAULT_POLICY, "wt_end_mode": 1.0}
    got = effective({"wt_out_s": 5}, base=base)
    assert got["wt_end_mode"] == 1.0 and got["wt_out_s"] == 5.0
    assert effective({"wt_out_s": 5})["wt_end_mode"] == 0.0          # no base: Bekky's defaults


def test_new_keys_default_to_bekky_behaviour():
    assert DEFAULT_POLICY["wt_end_mode"] == 0.0 and DEFAULT_POLICY["visits_source"] == 0.0
    assert DEFAULT_POLICY["wt_serve_s"] == 150.0 and DEFAULT_POLICY["wt_return_sim"] == 0.82


def test_policy_store_without_a_file_is_nothing_set_and_writes_nothing(tmp_path):
    # final review I3: the first load used to write the whole base, pinning every key against later yaml changes
    base = {**DEFAULT_POLICY, "visits_source": 1.0}
    store = policy_store(tmp_path)
    assert store.load() == {} and not (tmp_path / "policy.json").exists()
    assert effective(store.load(), base=base)["visits_source"] == 1.0
    store.save({"wt_out_s": 4.0})                                   # a PUT writes it; the hot reload sees it
    assert store.load() == {"wt_out_s": 4.0}


def test_the_zone_store_still_writes_its_default_on_first_load(tmp_path):
    from zones import zone_store
    assert zone_store(tmp_path).load() == {} and (tmp_path / "zones.json").read_text().strip() == "{}"


def test_register_source_accepts_2_and_switches_are_whole():
    assert validate_policy_update({"register_source": 2})["register_source"] == 2.0
    for bad in ({"register_source": 3}, {"wt_end_mode": 0.5}, {"visits_source": 2}, {"wt_return_sim": 1.2}):
        with pytest.raises(ValueError):
            validate_policy_update(bad)


def test_kids_car_keys_have_their_defaults_and_bounds():
    from policy import DEFAULT_POLICY, validate_policy_update
    assert DEFAULT_POLICY["kc_enabled"] == 0.0 and DEFAULT_POLICY["kc_every_s"] == 20.0
    for k in ("kc_every_s", "kc_confirm_n", "kc_lost_s", "kc_release_s", "kc_conf_min"):
        with pytest.raises(ValueError):
            validate_policy_update({k: 0})
    assert validate_policy_update({"kc_enabled": 1}) == {"kc_enabled": 1.0}


def test_per_chair_cover_keys_take_the_engines_chair_id():
    """Palm Hills' unpaired chairs are keyed by their zone id (area_of), e.g. ph3-L6, not the bare name."""
    assert validate_policy_update({"ce_carry_max_s@ph3-L6": 600, "ce_carry_lower_frac@ph2-L7": "1"}) == {
        "ce_carry_max_s@ph3-L6": 600.0, "ce_carry_lower_frac@ph2-L7": 1.0}
    with pytest.raises(ValueError, match=r"[<>]="):
        validate_policy_update({"ce_carry_lower_frac@ph3-L6": 0})


def test_re_id_thresholds_per_chair_keep_their_bounds_and_order():
    """2026-10-18 (user): the resume thresholds may differ per chair; low <= high on every chair, not only globally."""
    assert validate_policy_update({"ce_resume_sim_low@L6": 0.55, "ce_resume_sim_high@L6": 0.9}) == {
        "ce_resume_sim_low@L6": 0.55, "ce_resume_sim_high@L6": 0.9}
    with pytest.raises(ValueError, match="<="):
        validate_policy_update({"ce_resume_sim_high@L6": 1.2})
    validate_policy({**DEFAULT_POLICY, "ce_resume_sim_low@L6": 0.55, "ce_resume_sim_high@L6": 0.9})
    with pytest.raises(ValueError, match="L6"):
        validate_policy({**DEFAULT_POLICY, "ce_resume_sim_low@L6": 0.9})                # above the global high
    with pytest.raises(ValueError, match="L7"):
        validate_policy({**DEFAULT_POLICY, "ce_resume_sim_high@L7": 0.5})               # below the global low


def test_the_identity_check_after_cover_is_off_by_default():
    assert DEFAULT_POLICY["ce_reseat_check_s"] == 0.0
    assert validate_policy_update({"ce_reseat_check_s@L6": 30}) == {"ce_reseat_check_s@L6": 30.0}


def test_the_haircut_photo_keys_are_off_by_default_and_bounded():
    """2026-10-22 (spec haircut photos): a start and an end photo per haircut, off unless a site switches it on."""
    assert DEFAULT_POLICY["hp_photos"] == 0.0 and DEFAULT_POLICY["hp_end_every_s"] == 120.0
    assert (DEFAULT_POLICY["hp_crop_margin"], DEFAULT_POLICY["hp_crop_head"], DEFAULT_POLICY["hp_photo_px"]) == (0.25, 0.5, 640.0)
    assert validate_policy_update({"hp_photos": 1, "hp_end_every_s": 300, "hp_crop_margin": 0.4, "hp_crop_head": 1.5,
                                   "hp_photo_px": 800}) == {"hp_photos": 1.0, "hp_end_every_s": 300.0,
                                                            "hp_crop_margin": 0.4, "hp_crop_head": 1.5, "hp_photo_px": 800.0}
    for bad in ({"hp_photos": 0.5}, {"hp_photos": 2}, {"hp_end_every_s": 10}, {"hp_end_every_s": 4000},
                {"hp_crop_margin": 1.5}, {"hp_crop_head": 3}, {"hp_photo_px": 100}, {"hp_photo_px": 2000},
                {"hp_photo_px": 640.5}):
        with pytest.raises(ValueError):
            validate_policy_update(bad)
