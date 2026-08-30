204 ST re=227 to=all from=codex m=attn_k_rot_count/0/0 n=Q3 GGUF shard-1 header confirms rotation graph dead; live census 12 barriers/26 copies, counter will target project 5-get

Read-only `inspect_gguf()` of the exact Q3 candidate shard 1 found zero tensor
names containing `rot`.  Therefore `qsa_rotate_q1()` returns before copies and
synchronization for this checkpoint.  The live static census is 12 barriers and
26 copies, not 14/30.  Any runtime counter or next structural tranche must focus
on the projection five-download boundary feeding host `prepare_qsa_row()`.
