# YouTube Reference Research

The dashboard's `/research` page and `python main.py research --topic "gaming Shorts" --limit 5` use the official YouTube Data API to discover public metadata, then the official IFrame Player API for user-controlled viewing. By default, the UI is preview-only: it does not persist player timing, notes, or session observations, and it does not download, cache, extract, or analyze audiovisual streams.

Creative planning, music ranking, review, and revision run through **Puter.js in the dashboard browser** (free, keyless, user-pays; default model `qwen/qwen3.7-plus`, override with `PUTER_MODEL` in `.env`). No provider API keys exist in this project. The CLI has no browser and uses the measured local feature planner. Direct YouTube-URL video understanding is not used anywhere; the creative editor only ever sends low-resolution frame samples of your own footage to the model.

Files in this directory:

- `videos.jsonl`: reference metadata and review status. YouTube metadata expires after 30 days.
- `sessions.jsonl`: append-only player research-session state, created only if persistent observations have prior written approval.
- `observations.jsonl`: human-authored notes, created only if persistent observations have prior written approval.
- `metadata.json`: dataset version and record counts.

IFrame observations are marked `training_eligible: false` by default. They are not copied into `training/features/examples.jsonl` or consumed by the trainer unless the server is configured with `YOUTUBE_DERIVED_OBSERVATIONS_APPROVED=true`, the operator has confirmed creator/audio rights per observation, and the discovered license is Creative Commons. YouTube API policies restrict audiovisual copying and the creation of derived data from API data; obtain required written approval before enabling persistence. A Creative Commons search result is a discovery lead, not automatic clearance for every component or use.

For learning, provide independently obtained local references for which you have documented rights covering analysis and training. Use `ReferenceTrainingPipeline.import_local_video` with an accurate rights basis, license/permission details, and human annotations. Training remains creator-separated and refuses to promote a model with insufficient examples.

The research page binds to loopback by default. `YOUTUBE_DATA_API_KEY` is read server-side from `.env`; it is never returned to the browser. The IFrame Player API does not require this Data API key.