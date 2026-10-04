# Short-Form Reference Learning

## Online Research

Use `python main.py research --topic "gaming Shorts" --limit 5` to discover public metadata, or open `/research` in the local dashboard for configurable searches and the official YouTube embedded player. Research notes are human-authored and remain excluded from training. The app does not download or extract YouTube media. See [training/research/README.md](research/README.md) for storage, expiry, and policy details.

Only independently obtained, rights-cleared local media is eligible for the existing feature extraction and training pipeline. A YouTube `creativeCommon` search result does not itself grant API download access or establish that every component (including music) is reusable. Obtain all required rights and YouTube approvals for the intended use before importing local references.

Train the existing local-reference model with `python main.py train`. It reports `insufficient_references` when the dataset does not meet the trainer's minimum; player notes are never treated as training examples.

YouTube discovery and training media are separate. `YouTubeResearchAgent` calls the official YouTube Data API for public metadata (title, description, channel, URL, publication date, thumbnail URL, duration, category ID, reported license, and view count); it requests `videoLicense=creativeCommon` by default and does not download or extract YouTube media. Set `YOUTUBE_DATA_API_KEY` (or `YOUTUBE_API_KEY`) before running discovery. Search strategies cover multiple gaming topics and each discovery run caps results per channel. Recent `order=viewCount` searches are a trend-discovery proxy, not the official YouTube Trending feed, and view counts alone do not reveal editing techniques. A Creative Commons search result is a discovery lead, not a media download authorization through the API; obtain and verify training media separately and retain required attribution/license terms.

The creative agent may adapt broad patterns supported by multiple authorized references and the user's footage. It must not reproduce a reference video's sequence, exact timing, captions, or audio, or create a shot-for-shot replica.

Analyze only local videos for which the operator has confirmed one of `user_owned`, `licensed`, `public_domain`, or `explicitly_permitted`. Google Drive files can be imported from a locally synced/mounted Drive path. Import copies that media under `training/raw/<category>/` and writes separate metadata, analysis, feature, and timeline records plus `training/dataset.json`. Feature extraction currently measures duration, dimensions, FPS, codec, FFmpeg scene-cut density, shot duration, and audio level. Caption/effect/BGM annotations and moment-to-edit observations can be supplied to `import_local_video`; unknown signals remain null rather than being guessed.

Example:

```python
from pathlib import Path
from app.training.reference_pipeline import ReferenceTrainingPipeline

pipeline = ReferenceTrainingPipeline()
pipeline.import_local_video(
    Path("licensed_reference.mp4"),
    rights_basis="licensed",
    platform="youtube_shorts",
    style_tags=["competitive", "montage"],
    annotations={
        "features": {"caption_density": 0.4, "bgm_presence": True},
        "moments": [{
            "moment": {"visual_intensity": 0.9, "audio_intensity": 0.8},
            "observed_edit": {"zoom": 1.15, "cut": 1},
        }],
    },
)
```

Build the offline dataset and inspect it with `python training/build_dataset.py` and `python training/analyze_dataset.py`. Train and inspect an immutable candidate with `python training/train.py` and `python training/evaluate.py`. Splits are deterministic and grouped by creator to prevent leakage; the current target is 70/15/15 for train/validation/test. Promotion requires the validation score to beat a simple baseline and any active model. At least five distinct references must remain in training; single videos and duplicate hashes never define a style. User feedback is recorded in `training/feedback/edits.jsonl` and folded into the next candidate, not trained during an edit.

Collect a small candidate pool without acquiring files with `python training/collect.py --dry-run --license creativeCommon --order viewCount --query-limit 1 --max-results 5 --target 5`. The collector rotates through configured query groups, persists candidates and query state, and never downloads YouTube media. After independently obtaining media with permission, pass its local path and exact license/attribution metadata to `import_local_video`.

The dashboard's YouTube research is opt-in and no YouTube media download is part of this pipeline. When enabled, it defaults to two search strategies and five results per query to limit API quota. Configure `YOUTUBE_RESEARCH_QUERY_LIMIT`, `YOUTUBE_RESEARCH_RESULTS_PER_QUERY`, `YOUTUBE_RESEARCH_LOOKBACK_DAYS`, and `YOUTUBE_RESEARCH_INTERVAL_SECONDS` to adjust it. Model training remains an offline operation.