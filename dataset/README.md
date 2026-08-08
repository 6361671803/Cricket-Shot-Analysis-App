# Cricket shot training data

The app creates `shot_corrections.jsonl` when a user confirms or corrects a detected shot.
Those records are the verified labels for a future trained classifier.

For useful training, collect at least 100 clear full-body impact frames for every shot type, balanced across right- and left-handed batters and side, front, and diagonal camera views. Only add a player name when the player has agreed to it.

For the video-clip training data used by the future ML shot classifier (not the photo corrections above), see `../data/README.md` and `../label_clips.py` instead - that's a separate `data/<shot_name>/*.mp4` folder structure.
