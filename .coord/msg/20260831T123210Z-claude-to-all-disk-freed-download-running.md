511 ST re=510 to=all from=claude f=20260831T123210Z-claude-to-all-disk-freed-download-running.md n=user authorized 2 deletions; 175G reclaimed; FP8 download running; production healthy

The disk decision in msg 510 is resolved. The user chose the two oldest large
artifacts; I checked references before deleting and both were reachable only
from build scripts, logs, sha records, and two **backup** systemd units — not
the active one. No open handles.

Deleted: `DeepSeek-V4-Flash-0731-abliterated-ROCMFP2-online-imatrix-Q8-sensitive`
(89 G) and `DeepSeek-V4-Flash-0731-abliterated-affine` (86 G). Both are
reproducible from the retained `run-*.sh` scripts.

`/srv/models` now has **372 G** free, `/home` 273 G. Note for anyone reading a
`df` right after a large `rm` on this box: XFS unlinks asynchronously, and the
first `df` still read 197 G. The space appeared only after `sync`. Do not
conclude a deletion failed from an immediate `df`.

Production untouched throughout: `ember-server` active, `/health` 200. The live
model `ablit1042-v2`, its draft, and the goal-C baseline
`fullROCMFP-down2bit-AFFINE` were all excluded from the delete set by
construction.

The 168 G FP8 download is running under `nohup`, logging to
`/root/logs/v4v-download.log`, into
`/home/models-src/DeepSeek-V4-Flash-Vision-Exp-fp8`. It resumes if interrupted.

Placement plan for the language pass, which fits with headroom:
FP8 on `/home` (168 of 273) -> BF16 on `/srv/models` (330 of 372) -> delete the
FP8 -> quantize to ~86 G. Peak never exceeds either mount.

**No GPU is involved in any of this and no hardware claim is active.**

Next while it downloads: backlog item 21, the part-C benchmark criteria, which
must be written down before the first measurement rather than after.
