"""Sample complete training video waves before rendering or transferring frames.

RLinf calls ``flush_video`` after each rollout wave. The historical setting
records every wave. ``record_every_n_episodes`` selects waves 0, N, 2N, ...;
each wave is one tiled batch of environment episodes, not one vector slot.
"""

import numbers
from pathlib import Path


class SampledVideoMixin:
    """Gate native RecordVideo frame collection while retaining its lifecycle."""

    def __init__(self, env, video_cfg, fps=None):
        interval = video_cfg.get("record_every_n_episodes", 1)
        if (
            isinstance(interval, bool)
            or not isinstance(interval, numbers.Integral)
            or interval < 1
        ):
            raise ValueError("record_every_n_episodes must be a positive integer")
        self.record_every_n_episodes = int(interval)
        self._video_wave_touched = False
        super().__init__(env, video_cfg, fps=fps)

    def _records_current_wave(self) -> bool:
        return self.video_cnt % self.record_every_n_episodes == 0

    def add_new_frames(self, obs, infos=None, rewards=None, terminations=None):
        """Skip observation extraction, capture_image and overlays on omitted waves."""
        self._video_wave_touched = True
        if self._records_current_wave():
            return super().add_new_frames(obs, infos, rewards, terminations)
        return None

    def record_video_in_result(self, result) -> None:
        """Skip even terminal-observation processing for omitted chunk results."""
        self._video_wave_touched = True
        if self._records_current_wave():
            return super().record_video_in_result(result)
        return None

    def flush_video(self, video_sub_dir=None):
        """Keep original wave numbers in sparse filenames; ignore duplicate flushes."""
        if self.record_every_n_episodes == 1:
            return super().flush_video(video_sub_dir)
        if not self._video_wave_touched:
            return None
        if self.render_images:
            # Native flush writes <video_cnt>.mp4 and advances the counter once.
            result = super().flush_video(video_sub_dir)
        else:
            # Omitted waves collect no frames but must advance original numbering.
            self.video_cnt += 1
            result = None
        self._video_wave_touched = False
        return result


def install_video_overlay(wrapper_root: Path, native_source: Path) -> Path:
    """Install a launch-local wrapper which Ray workers can import independently.

    The native file is copied unchanged under another module name; its source
    installation is never modified. Other wrapper modules remain symlinks.
    """
    source = native_source.read_text()
    compile(source, str(native_source), "exec")
    (wrapper_root / "_record_video_native.py").write_text(source)
    (wrapper_root / "_spatial_wovr_video.py").write_text(Path(__file__).read_text())
    target = wrapper_root / "record_video.py"
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(target)
    target.write_text(
        '"""Launch-local sparse recording; default native behavior is retained."""\n'
        "from rlinf.envs.wrappers._record_video_native import RecordVideo as NativeRecordVideo\n"
        "from rlinf.envs.wrappers._spatial_wovr_video import SampledVideoMixin\n"
        "\n\nclass RecordVideo(SampledVideoMixin, NativeRecordVideo):\n"
        '    """Collect only configured rollout waves, before capture/render work."""\n'
    )
    return target
