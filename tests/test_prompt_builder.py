import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.prompt_builder import (
    H3_CONTEXT_LEAD_IN_FRAMES,
    I2V_TEMPLATE,
    MINIMAX_BASE_PROMPT,
    MINIMAX_REF_CONTEXT_INCOMING,
    MINIMAX_REF_CONTEXT_OUTGOING,
    MINIMAX_REF_PROMPT,
    R2V_TEMPLATE,
    T2V_TEMPLATE,
    build_context_handoff_guide,
    build_prompt_request,
    get_system_prompt_options,
)


def test_system_prompt_options_include_mode_and_image_rules():
    options = get_system_prompt_options()

    assert {
        "key": "default_t2v",
        "task_type": "t2v",
        "mode": "default",
        "min_images": 0,
        "max_images": 0,
        "system_prompt": T2V_TEMPLATE,
    } in options
    assert {
        "key": "default_i2v",
        "task_type": "i2v",
        "mode": "default",
        "min_images": 0,
        "max_images": None,
        "system_prompt": I2V_TEMPLATE,
    } in options


def test_system_prompt_options_include_ref_template_without_image_filtering():
    options = get_system_prompt_options()

    assert {
        "key": "ref_r2v",
        "task_type": "r2v",
        "mode": "ref",
        "min_images": 0,
        "max_images": None,
        "system_prompt": R2V_TEMPLATE,
    } in options


def test_system_prompt_options_include_minimax_mode_groups():
    options = get_system_prompt_options()

    assert {
        "key": "minimax_base",
        "format": "MiniMax",
        "modes": ["default", "l2v"],
        "system_prompt": MINIMAX_BASE_PROMPT,
    } in options
    assert {
        "key": "minimax_ref",
        "format": "MiniMax",
        "modes": ["ref", "edit"],
        "system_prompt": MINIMAX_REF_PROMPT,
    } in options


def test_custom_system_prompt_preserves_unknown_braced_text():
    custom_template = 'Write JSON like {"subject": "{character}"}. Prompt: {user_prompt}'

    _, prompt, _ = build_prompt_request(
        "v2v",
        "make it move",
        custom_system_prompt=custom_template,
    )

    assert prompt == 'Write JSON like {"subject": "{character}"}. Prompt: make it move'


def test_minimax_uses_mode_specific_system_prompts():
    for task_type in ("t2v", "i2v", "l2v"):
        system_prompt, user_prompt, json_mode = build_prompt_request(
            task_type, "make a video", video_format="MiniMax"
        )
        assert (system_prompt, user_prompt, json_mode) == (
            MINIMAX_BASE_PROMPT, "make a video", False
        )

    for task_type in ("r2v", "rv2v", "v2v", "vi2v"):
        system_prompt, user_prompt, json_mode = build_prompt_request(
            task_type, "change the video", video_format="MiniMax"
        )
        assert (system_prompt, user_prompt, json_mode) == (
            MINIMAX_REF_PROMPT, "change the video", False
        )

    system_prompt, _, _ = build_prompt_request(
        "custom-task",
        "make a video",
        video_format="MiniMax",
        task_mode="default",
    )
    assert system_prompt == MINIMAX_BASE_PROMPT


def test_context_handoff_guide_is_empty_without_a_continuation():
    assert build_context_handoff_guide() == ""
    assert build_context_handoff_guide(False, False) == ""


def test_context_handoff_guide_incoming_only():
    guide = build_context_handoff_guide(continues_previous=True)

    assert guide.startswith("## 8. Continuation From a Previous Segment")
    assert "## 9." not in guide
    assert "0.92 seconds" in guide
    assert "{lead_in}" not in guide
    assert "00:02.000" in guide


def test_context_handoff_guide_outgoing_only():
    guide = build_context_handoff_guide(continued_by_next=True)

    assert guide == MINIMAX_REF_CONTEXT_OUTGOING
    assert "## 8." not in guide


def test_context_handoff_guide_both_directions_keeps_incoming_first():
    guide = build_context_handoff_guide(continues_previous=True, continued_by_next=True)

    assert guide.index("## 8.") < guide.index("## 9.")


def test_context_handoff_guide_lead_in_follows_frame_rate():
    guide = build_context_handoff_guide(continues_previous=True, frame_rate=30)

    assert "0.73 seconds" in guide
    assert "0.92 seconds" not in guide


def test_context_lead_in_frames_match_the_project_node():
    source = (Path(__file__).resolve().parents[1] / "nodes" / "project.py").read_text()
    match = re.search(r"^H3_CONTEXT_SOURCE_FRAMES\s*=\s*(\d+)", source, re.MULTILINE)

    assert match is not None
    assert H3_CONTEXT_LEAD_IN_FRAMES == int(match.group(1))


def test_minimax_reference_prompt_appends_context_handoff_guide():
    expected_guide = build_context_handoff_guide(True, True)

    system_prompt, user_prompt, json_mode = build_prompt_request(
        "r2v",
        "she keeps walking",
        video_format="MiniMax",
        task_mode="ref",
        continues_previous=True,
        continued_by_next=True,
    )

    assert system_prompt == f"{MINIMAX_REF_PROMPT}\n\n{expected_guide}"
    assert (user_prompt, json_mode) == ("she keeps walking", False)


def test_minimax_reference_prompt_is_unchanged_without_continuation_flags():
    system_prompt, _, _ = build_prompt_request(
        "r2v", "a prompt", video_format="MiniMax", task_mode="ref"
    )

    assert system_prompt == MINIMAX_REF_PROMPT


def test_context_handoff_guide_only_extends_reference_generation():
    flags = {"continues_previous": True, "continued_by_next": True}

    base_system, _, _ = build_prompt_request(
        "i2v", "a prompt", video_format="MiniMax", task_mode="default", **flags
    )
    l2v_system, _, _ = build_prompt_request(
        "l2v", "a prompt", video_format="MiniMax", task_mode="l2v", **flags
    )
    edit_system, _, _ = build_prompt_request(
        "rv2v", "a prompt", video_format="MiniMax", task_mode="edit", **flags
    )
    custom_system, _, _ = build_prompt_request(
        "r2v",
        "a prompt",
        video_format="MiniMax",
        task_mode="ref",
        custom_system_prompt="My own rules",
        **flags,
    )

    assert base_system == MINIMAX_BASE_PROMPT
    assert l2v_system == MINIMAX_BASE_PROMPT
    assert edit_system == MINIMAX_REF_PROMPT
    assert custom_system == "My own rules"


def test_context_handoff_guide_falls_back_to_task_type_without_task_mode():
    flags = {"continues_previous": True}

    reference_system, _, _ = build_prompt_request(
        "r2v", "a prompt", video_format="MiniMax", **flags
    )
    editing_system, _, _ = build_prompt_request(
        "rv2v", "a prompt", video_format="MiniMax", **flags
    )

    assert reference_system.endswith(MINIMAX_REF_CONTEXT_INCOMING.replace("{lead_in}", "0.92"))
    assert editing_system == MINIMAX_REF_PROMPT
