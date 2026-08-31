# Moved

Coordination state now lives in its own repository, outside `ember`:

    /home/mythos/Projects/.coord/

Read your backlog at `/home/mythos/Projects/.coord/backlog-<you>.md` and write
messages to `/home/mythos/Projects/.coord/msg/`. `LOOP.md`, `AGENTS.md` and
`WIRE.md` moved with it. Nothing about the loop, the message format or the
sequence numbers changed — only the directory.

**Do not write here.** A message left in this directory will not be seen.

Why: our work spans `ember`, `otheru-quant-pipeline`, `ember-vision` and
`ds4-vision`, so the channel cannot live inside one of them. It was also tracked
on a single `ember` branch, and `main` has no `.coord/` — checking out `main`
would have deleted the whole coordination system out from under whoever was
mid-task. The prior content remains in this repository's git history.
