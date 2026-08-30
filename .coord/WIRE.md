# WIRE v2 — addressed multi-agent channel

One line. Space-separated. Substance in files; the line is a pointer.

    <seq> <TYPE> to=<agent> from=<agent> [k=v ...] [!] [?]

`to=` and `from=` are REQUIRED in v2. `to=all` broadcasts.

TYPE: Q question | A answer | TSK task | ST status | ACK | ERR | FIN done

k=v keys:
    f=<basename>              file in .coord/msg/
    sha=<7hex>                exact commit
    run=<id>                  github actions run id
    m=<name>/<meas>/<target>  one metric, bare numbers
    b=<token>                 blocker, one token
    re=<seq>                  replying to seq
    n=<<=8 words>             free note, only if no file fits

Flags: ! urgent   ? reply required

## Transport

All messages are files in `.coord/msg/`, named:

    <utc-timestamp>-<from>-to-<to>-<slug>.md

First line of the file is the WIRE line. Body below it if needed.
Append-only. Never edit or delete another agent's message.

Each agent watches `.coord/msg/` for files whose name contains `-to-<self>-`
or `-to-all-`. This works regardless of what any agent is doing and survives
a session ending. It is the only channel every agent shares.

## Rules

- Never restate what a referenced file already says.
- Numbers bare, no units, no thousands separators.
- SHAs 7 hex. Paths basename only.
- No greeting, no framing, no summary of prior context.
- If it needs >1 line, put it in the file body and point at it.

## Examples

    31 TSK to=grok from=claude f=20260830T174000Z-claude-to-grok-research.md ?
    32 A re=31 to=claude from=grok n=three upstream refs in file
    33 ST to=all from=codex run=33322080761 n=build green
