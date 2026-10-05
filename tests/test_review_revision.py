import pathlib

import app.agent.short_form_editor as editor_module
from app.agent.short_form_editor import ShortFormCreativeEditor
from app.editing.editor import EditOptions


def test_revised_render_is_reviewed_and_stores_final_critique(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATIVE_REVIEW_RENDER", "true")
    source = tmp_path / "gameplay.mp4"
    output = tmp_path / "result.mp4"
    source.write_bytes(b"source")
    raw_plan = {
        "platform": "youtube_shorts",
        "strategy": "continuous",
        "target_duration": 2,
        "shots": [{"source_index": 0, "start": 0, "end": 2, "role": "action", "transition": "cut"}],
        "music_requirements": {},
        "sound_design": [],
    }

    class Model:
        review_count = 0

        def create_plan(self, **kwargs):
            return raw_plan

        def review_render(self, plan, context, frames):
            self.review_count += 1
            if self.review_count == 1:
                return {"needs_revision": True, "critique": "Crop needs adjustment.", "revision_request": "Fill the frame."}
            return {"needs_revision": False, "critique": "Revised crop passes.", "revision_request": ""}

        def revise_plan(self, **kwargs):
            return raw_plan

    class Style:
        def learn(self, platform):
            return {"training_status": "no_promoted_model"}

    class Sfx:
        def list_assets(self):
            return []

    renders = []

    def analyze(paths, max_sources=12):
        return {"sources": [{
            "source_index": 0, "filename": pathlib.Path(paths[0]).name,
            "duration": 2, "width": 1080, "height": 1920, "has_audio": True,
            "audio_mean_db": -30,
        }]}, [{"time": 0.5, "data_url": "data:image/jpeg;base64,eA=="}]

    def render(inputs, destination, options, callback):
        renders.append(destination)
        pathlib.Path(destination).write_bytes(b"rendered")
        return pathlib.Path(destination)

    monkeypatch.setattr(editor_module, "analyze_sources", analyze)
    model = Model()
    artifact = ShortFormCreativeEditor(
        model=model,
        music_provider=type("Music", (), {"client_id": ""})(),
        style_learner=Style(),
        sfx_library=Sfx(),
        renderer=render,
    ).create_edit(
        [source], output,
        EditOptions(creative_mode=True, platform="youtube_shorts", aspect_ratio="9:16", target_duration=2, bgm_track=None),
    )

    assert len(renders) == 2
    assert model.review_count == 2
    assert artifact["revision_count"] == 1
    assert artifact["review"]["needs_revision"] is False
    assert artifact["review"]["critique"] == "Revised crop passes."
    assert artifact["review"]["previous_critique"] == "Crop needs adjustment."