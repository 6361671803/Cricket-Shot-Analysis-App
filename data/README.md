# Training clip data

This folder is the input for the future ML shot classifier (`classify_shot_ml`,
not built yet - see the Phase 3 plan). It's created and populated by you,
using `label_clips.py` in the project root, or by hand.

## Rights: self-recorded footage only

Only put footage here that you recorded yourself or otherwise have the
rights to use for training. **Do not use broadcast match footage** (IPL,
international matches, any TV/streaming coverage) - that's copyrighted and
isn't eligible for this. Practice sessions, nets footage, or footage from
players who've agreed to it are the intended source.

## Folder structure

```
data/
  <shot_name>/
    clip_001.mp4
    clip_002.mp4
    ...
```

`<shot_name>` must exactly match one of the keys in `../shots.json`
(case-sensitive) - run this to see the current full list:

```
python -c "import shots; print(list(shots.SHOT_DB.keys()))"
```

`label_clips.py` creates these folders for you automatically as you label
clips, so you don't need to pre-create them.

## How many clips do you need?

This depends on which classifier Phase 3 ends up training:

- **RandomForest on hand-crafted features** (the planned default - joint
  angles/velocities extracted from the pose sequence, not raw pixels):
  workable from as few as **~30-50 clips per class**, meaningfully better
  by **~100+ per class**. This is realistic for one person to self-film.
- **A small LSTM/CNN on raw sequences**: needs substantially more data,
  realistically **several hundred+ clips per class**, to avoid overfitting.
  Not practical to hand-film alone for all 24 classes - only worth pursuing
  once a RandomForest baseline is working and more data has accumulated
  (e.g. from real users via the in-app "Is this the correct shot?"
  correction flow - see `../dataset/README.md`).

Don't feel you need every class covered before anything can be trained -
a classifier can start with a subset of shots and grow as more clips come
in.

## Recording guidelines

- 120fps+ slow-motion preferred, matching the rest of the app's assumptions.
- Keep the full body in frame for the whole shot (stance through
  follow-through).
- Record both right- and left-handed batters where possible - a classifier
  trained only on one stance will generalize poorly to the other. (Mirroring
  existing clips horizontally is a cheaper alternative worth considering
  later, but isn't implemented yet.)
- Vary camera angle (side-on, front-on, diagonal) similar to the app's own
  A/B/C angle options, so the classifier isn't accidentally learning to
  recognize a single camera setup instead of the shot itself.

## Labeling your footage

Use `label_clips.py` (project root) to step through a longer recording
(e.g. a full nets session) and cut out individual labeled clips:

```
python label_clips.py path/to/your_recording.mp4
```

See the script's own header comment for the keyboard controls. It writes
trimmed clips straight into `data/<shot_name>/`, validating the shot name
against `shots.json` so you can't accidentally create a typo'd class.
