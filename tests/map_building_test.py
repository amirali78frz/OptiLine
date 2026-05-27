"""
map_building_test.py
====================
Tests and demonstrations for OptiLine's MapBuilder class.

Run from the project root:
    python tests/map_building_test.py

Or run individual stages by commenting others out at the bottom.
"""

import os
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running from both the project root and from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from OptiLine.map_builder import (
    MapBuilder,
    generate_random_track,
    build_examples,
    TRACK_STYLES,
    _arc_lengths,
)

# ===========================================================================
#  Configuration
# ===========================================================================

SEED          = 42
N_GALLERY     = 5       # tracks shown in the gallery plot (Stage 1)
N_BATCH       = 8       # tracks generated to disk in the batch test (Stage 2)
TARGET_LENGTH = 450.0   # metres


# ===========================================================================
#  Stage 1 — Single-track generation and gallery plot
# ===========================================================================

def stage1_gallery():
    """
    Generate one track per style and display them side by side.
    No file I/O — pure geometry.
    """
    print("\n" + "=" * 60)
    print("Stage 1: per-style gallery (no file I/O)")
    print("=" * 60)

    mb = MapBuilder(seed=SEED, target_length=TARGET_LENGTH)

    fig, axes = plt.subplots(1, len(TRACK_STYLES), figsize=(4 * len(TRACK_STYLES), 4))
    fig.suptitle("Stage 1 — One track per style  (seed={})".format(SEED), fontsize=13)

    stats = []
    for ax, style in zip(axes, TRACK_STYLES):
        t0    = time.perf_counter()
        track = mb.generate_track(style=style)
        dt    = time.perf_counter() - t0

        xy     = track["xy"]
        w_r    = track["w_right"]
        w_l    = track["w_left"]
        length = _arc_lengths(np.vstack([xy, xy[0]]))[-1]

        # Compute boundary edges for display
        dp      = np.vstack([np.diff(xy, axis=0), xy[0] - xy[-1]])
        seg_len = np.hypot(dp[:, 0], dp[:, 1]) + 1e-12
        nx      = -dp[:, 1] / seg_len
        ny      =  dp[:, 0] / seg_len
        right_e = xy - np.column_stack([nx * w_r, ny * w_r])
        left_e  = xy + np.column_stack([nx * w_l,  ny * w_l])

        ax.fill(
            np.concatenate([left_e[:, 0], right_e[::-1, 0]]),
            np.concatenate([left_e[:, 1], right_e[::-1, 1]]),
            color="lightgrey", zorder=0
        )
        ax.plot(np.append(left_e[:,  0], left_e[0,  0]),
                np.append(left_e[:,  1], left_e[0,  1]), "k-", lw=0.8)
        ax.plot(np.append(right_e[:, 0], right_e[0, 0]),
                np.append(right_e[:, 1], right_e[0, 1]), "k-", lw=0.8)
        ax.plot(np.append(xy[:, 0], xy[0, 0]),
                np.append(xy[:, 1], xy[0, 1]), "b--", lw=0.6, alpha=0.5)
        ax.plot(xy[0, 0], xy[0, 1], "go", ms=6, label="start")
        ax.set_title(f"{style}\n{length:.0f} m  |  {len(xy)} pts", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

        stats.append((style, len(xy), length, dt * 1000))
        print(f"  {style:<20s}  pts={len(xy):4d}  length={length:5.0f}m  "
              f"time={dt*1000:.1f}ms")

    plt.tight_layout()
    plt.show()
    return stats


# ===========================================================================
#  Stage 2 — Batch generation to disk (temporary directory)
# ===========================================================================

def stage2_batch_to_disk():
    """
    Generate N_BATCH maps into a temporary directory and verify the files.
    """
    print("\n" + "=" * 60)
    print(f"Stage 2: batch generation ({N_BATCH} maps) → temp directory")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        mb      = MapBuilder(seed=SEED + 1, variable_width=False)
        t0      = time.perf_counter()
        folders = mb.generate(n=N_BATCH, start_index=1, maps_dir=tmp, verbose=True)
        dt      = time.perf_counter() - t0

        print(f"\n  Total generation time: {dt:.2f}s  "
              f"({dt / N_BATCH * 1000:.0f} ms/map)")

        # Verify files
        n_ok = 0
        for folder in folders:
            name     = os.path.basename(folder)
            expected = [
                f"{name}_centerline.csv",
                f"{name}_map.png",
                f"{name}_map.yaml",
            ]
            missing = [f for f in expected if not os.path.isfile(os.path.join(folder, f))]
            if missing:
                print(f"  [FAIL] {name}: missing {missing}")
            else:
                n_ok += 1

        print(f"\n  File check: {n_ok}/{N_BATCH} examples complete  ✓")
        assert n_ok == N_BATCH, f"Expected {N_BATCH} complete examples, got {n_ok}"

    return dt


# ===========================================================================
#  Stage 3 — Variable-width tracks
# ===========================================================================

def stage3_variable_width():
    """
    Generate 3 variable-width tracks and compare width profiles.
    """
    print("\n" + "=" * 60)
    print("Stage 3: variable-width tracks")
    print("=" * 60)

    mb = MapBuilder(seed=SEED + 2, variable_width=True, w_min=0.6, w_max=2.5)

    n_tracks = 3
    fig, axes = plt.subplots(2, n_tracks, figsize=(5 * n_tracks, 7))
    fig.suptitle("Stage 3 — Variable-width tracks", fontsize=13)

    for col in range(n_tracks):
        track  = mb.generate_track()
        xy     = track["xy"]
        w_r    = track["w_right"]
        w_l    = track["w_left"]
        style  = track["style"]
        length = _arc_lengths(np.vstack([xy, xy[0]]))[-1]

        # Arc-length parameter
        s = _arc_lengths(xy)

        # Top: track layout
        ax_top = axes[0, col]
        dp      = np.vstack([np.diff(xy, axis=0), xy[0] - xy[-1]])
        seg_len = np.hypot(dp[:, 0], dp[:, 1]) + 1e-12
        nx      = -dp[:, 1] / seg_len
        ny      =  dp[:, 0] / seg_len
        right_e = xy - np.column_stack([nx * w_r, ny * w_r])
        left_e  = xy + np.column_stack([nx * w_l,  ny * w_l])

        ax_top.fill(
            np.concatenate([left_e[:, 0], right_e[::-1, 0]]),
            np.concatenate([left_e[:, 1], right_e[::-1, 1]]),
            color="lightgrey"
        )
        ax_top.plot(np.append(left_e[:,  0], left_e[0,  0]),
                    np.append(left_e[:,  1], left_e[0,  1]), "k-", lw=0.8)
        ax_top.plot(np.append(right_e[:, 0], right_e[0, 0]),
                    np.append(right_e[:, 1], right_e[0, 1]), "k-", lw=0.8)
        ax_top.set_title(f"{style}\n{length:.0f} m", fontsize=9)
        ax_top.set_aspect("equal")
        ax_top.axis("off")

        # Bottom: width profile
        ax_bot = axes[1, col]
        ax_bot.plot(s, w_r, label="right", color="tab:blue")
        ax_bot.plot(s, w_l, label="left",  color="tab:orange")
        ax_bot.fill_between(s, 0, w_r, alpha=0.15, color="tab:blue")
        ax_bot.fill_between(s, 0, w_l, alpha=0.15, color="tab:orange")
        ax_bot.set_xlabel("Arc length (m)")
        ax_bot.set_ylabel("Half-width (m)")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

        print(f"  Track {col+1}: {style:<20s}  length={length:.0f}m  "
              f"w_right=[{w_r.min():.2f},{w_r.max():.2f}]m  "
              f"w_left=[{w_l.min():.2f},{w_l.max():.2f}]m")

    plt.tight_layout()
    plt.show()


# ===========================================================================
#  Stage 4 — Reproducibility check
# ===========================================================================

def stage4_reproducibility():
    """
    Verify that two MapBuilder instances with the same seed produce identical
    tracks and that reset() restores the sequence.
    """
    print("\n" + "=" * 60)
    print("Stage 4: reproducibility")
    print("=" * 60)

    # Two independent builders with the same seed → must be identical
    mb1 = MapBuilder(seed=SEED)
    mb2 = MapBuilder(seed=SEED)

    tracks1 = [mb1.generate_track() for _ in range(5)]
    tracks2 = [mb2.generate_track() for _ in range(5)]

    all_equal = all(
        np.allclose(t1["xy"], t2["xy"]) and
        np.allclose(t1["w_right"], t2["w_right"])
        for t1, t2 in zip(tracks1, tracks2)
    )
    print(f"  Two builders, same seed → identical: {'✓' if all_equal else '✗ FAIL'}")
    assert all_equal, "Reproducibility check failed!"

    # reset() restores the sequence
    mb1.reset()
    tracks1b = [mb1.generate_track() for _ in range(5)]
    after_reset = all(
        np.allclose(t1["xy"], t2["xy"])
        for t1, t2 in zip(tracks1, tracks1b)
    )
    print(f"  reset() restores sequence     → identical: {'✓' if after_reset else '✗ FAIL'}")
    assert after_reset, "reset() reproducibility check failed!"

    # Different seed → must differ
    mb3    = MapBuilder(seed=SEED + 99)
    diff   = not np.allclose(mb3.generate_track()["xy"], tracks1[0]["xy"])
    print(f"  Different seed → different xy : {'✓' if diff else '✗ (suspiciously equal)'}")

    print("  All reproducibility checks passed.")


# ===========================================================================
#  Stage 5 — Statistics across a larger batch (no file I/O)
# ===========================================================================

def stage5_statistics(n: int = 50):
    """
    Generate *n* tracks without file I/O and plot length / style distributions.
    """
    print("\n" + "=" * 60)
    print(f"Stage 5: statistics over {n} generated tracks")
    print("=" * 60)

    mb      = MapBuilder(seed=SEED + 3, target_length=TARGET_LENGTH)
    lengths = []
    styles  = []

    for _ in range(n):
        track  = mb.generate_track()
        xy     = track["xy"]
        length = _arc_lengths(np.vstack([xy, xy[0]]))[-1]
        lengths.append(length)
        styles.append(track["style"])

    lengths = np.array(lengths)
    print(f"  length  mean={lengths.mean():.1f}m  "
          f"std={lengths.std():.1f}m  "
          f"min={lengths.min():.1f}m  max={lengths.max():.1f}m")

    style_counts = {s: styles.count(s) for s in TRACK_STYLES}
    print("  style distribution:", style_counts)

    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Stage 5 — Statistics over {n} tracks  (seed={SEED+3})", fontsize=12)

    ax_l.hist(lengths, bins=15, color="steelblue", edgecolor="white")
    ax_l.axvline(lengths.mean(), color="red", ls="--", label=f"mean={lengths.mean():.0f}m")
    ax_l.set_xlabel("Track length (m)")
    ax_l.set_ylabel("Count")
    ax_l.set_title("Length distribution")
    ax_l.legend()
    ax_l.grid(True, alpha=0.3)

    ax_s.bar(style_counts.keys(), style_counts.values(), color="teal", edgecolor="white")
    ax_s.set_xlabel("Style")
    ax_s.set_ylabel("Count")
    ax_s.set_title("Style frequency")
    ax_s.tick_params(axis="x", rotation=20)
    ax_s.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    return lengths, style_counts


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    stage1_gallery()
    stage2_batch_to_disk()
    stage3_variable_width()
    stage4_reproducibility()
    stage5_statistics(n=50)

    print("\n" + "=" * 60)
    print("All map_building_test stages completed successfully.")
    print("=" * 60)
