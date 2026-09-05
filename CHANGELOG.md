# Changelog

Ember uses calendar versions in `YEAR.MONTH.DAY` form without zero-padding.
Release notes record user-visible features, fixes, compatibility changes,
upgrade steps, and validated hardware. Git tags add a `v` prefix; for example,
version `2026.8.10` is tagged `v2026.8.10`.

`VERSION` is authoritative. Ember publishes at most one release per calendar
day; additional fixes remain on `main` until the next dated release rather than
using an ambiguous same-day suffix.

## Unreleased

### Added

- **vision:** native 8-bit grayscale and RGB/YCbCr JPEG decoding, including
  progressive images, through libjpeg-turbo. Complete-file preflight bounds
  dimensions, pixels, input size and scans; codec warnings reject partial
  images. CMYK/non-8-bit JPEG, WebP and GIF remain unsupported. EXIF rotation
  and ICC transforms are not applied. GPU request validation remains a release
  gate; host tests cover decoded pixels and shared PNG/JPEG preprocessing.

## 2026.9.3

### Curated notes

Ember becomes a vision engine. Its default model is now the vision-capable
DeepSeek-V4-Flash-Vision-Exp, and the container builds on ROCm 10.0.

### Added

- **vision:** native DeepSeek-V4 image support. Bounded PNG decoder, image
  preprocessing reproduced against the reference implementation, native tower
  execution and contract loader, learned prefill graph, and image-token routing
  through the model's second router bias (`exp_probs_b_vl`) with in-span
  bidirectional attention visibility. Images are PNG only; JPEG, WebP and GIF
  are refused rather than handed to a permissive decoder.
- **vision:** a two-armed behavioural gate. Arm B sends each question with no
  image, and any item it answers is cut rather than scored, so a model that
  ignores images cannot pass on text priors.
- **deepseek4:** importance-matrix collection inside the engine
  (`DFLASH_IMATRIX_OUT`), covering both the score-routed and hash-routed MoE
  paths and writing the legacy `.dat` the affine quantizer consumes. Upstream's
  collector has no image path for any model, so this is the only way to
  calibrate a quantization on the routing images actually take.
- **entrypoint:** the vision tower is downloaded and verified alongside the
  model and the drafter. A missing tower is a hard error, not a quiet fall back
  to the text path.

### Changed

- **scope:** the unfinished second-architecture runtime, its CI lanes and its
  documentation are no longer carried on `main`. They were never released and
  the fourteen workflows they brought fired on every push; the work continues
  on its own branch. Nothing shipped changes.
- **container:** ROCm 10.0.0, pinned by digest, with a build-time assertion that
  the toolchain reports that version.
- **entrypoint:** the default DeepSeek deployment is the vision model. The
  text-only 0731 artifact is neither deprecated nor deleted and stays servable
  through `EMBER_MODEL_REPO` and `EMBER_MODEL_REVISION`.

### Fixed

- **deepseek4:** vision support had put a cross-translation-unit call inside the
  prefill causal-mask loop, where it could neither inline nor vectorise. It cost
  6x on that loop for text requests, which contain no images at all.
- **deepseek4:** four defects in the imatrix collector, each of which would have
  produced a well-formed file with wrong or missing contents: a graph-output
  flag set on a view rather than its base, registration on graph paths that
  never drain, the layer-major graph cache silently skipping collection, and an
  empty matrix that passed its own coverage check.
- **engine:** layer capture bound to its authority bundle; single-layer control
  model admission.

### Performance

- **deepseek4:** the prefill embed buffer and the exact-prefill attention
  metadata arena are reused across chunks instead of reallocated.
- **rocmfp2/rocmfp4:** an identity `v_perm_b32` removed from the FP2 unpack. Two
  further changes, UE4M3 scale decode through the f16 converter and the FP2
  affine offset term from `block_q8_1.ds.y`, are compiled out by default and
  carry no throughput claim.

### Measured and rejected

- The flash-attention D=512 q-rope-tail prepass is not in this release. It
  removed 2,710 instructions and 258 flat loads from the kernel and made prefill
  slower: p50 63.80 ms to 84.73 ms on the same kernel, count and grid.
- Both ROCm runtime environment overrides were measured and dropped.
  `DEBUG_CLR_DIRECT_DOORBELL=1` hung a request outright.

### Not measured

Vision throughput at the serving expert top-k, and quantization quality against
BF16 for either model. Neither has a number, and the model cards say so.

### Added

- **deepseek4:** register the hash-routed layers with the imatrix collector (`628ff503`)
- **deepseek4:** collect the routed-expert imatrix in the engine (`fd3f2fe5`)
- **deepseek4:** activation dump entry point for refusal-direction work (`4f8dea38`)
- **vision:** land the behavioural gate and the build revision guard (`125d2498`)
- **vision:** report tower encode timing (`5df70f61`)
- **vision:** expose single-layer server control (`01b78008`)
- **vision:** wire native request tower (`770d2b2f`)
- **vision:** reproduce native image preprocessing (`83914b67`)
- **vision:** add bounded PNG decoder (`98326850`)
- **vision:** bind image encoding to prompt offsets (`75f76618`)
- **engine:** add bound layer-0 capture probe (`f8765103`)
- **engine:** add bound full-logit control probe (`4158e87a`)
- **engine:** add native vision tower execution (`8ac6d03f`)
- **engine:** add native vision tower contract loader (`952d21f1`)
- **vision:** add grounded offline language gate (`ca1055df`)
- **vision:** replace palette prefill runtime (`43fa6ce3`)
- **vision:** wire learned prefill graph (`9b4f532b`)
- **vision:** validate learned prefill runs (`0192b872`)
- **vision:** expand exact image placeholders (`609accea`)
- **vision:** generalize request image runs (`95ff31b3`)
- **vision:** expand learned image placeholders (`b8e48cc0`)
- **vision:** render DeepSeek image placeholders (`f94d6820`)
- **vision:** load vision router bias sets (`a5ed586b`)
- **vision:** keep learned image blocks atomic (`f16a01f9`)
- **vision:** load learned mmproj markers (`3bebb6df`)
- **vision:** load bounded aligner artifacts (`67c9d1b4`)
- **vision:** version offline aligner artifacts (`464bfbf4`)
- **vision:** assemble learned image embedding blocks (`86ebb400`)
- **vision:** pin learned marker language contract (`cdef693c`)
- **vision:** normalize inline media and contain cache keys (`55a4d732`)
- **server:** allow operator-forced exact prefill (`ecf6996b`)
- **engine:** inventory MMQ activation layouts (`3cf33109`)
- **scripts:** convert third-party Qwen MTP GGUFs into Ember's companion format (`b887ca3a`)
- **engine:** accept Q4_K expert matrices (`5ac6d956`)
- **engine:** gate prefill divergence by logit margin (`01b82184`)
- **engine:** capture GDN recurrence inputs (`7a09594f`)
- **engine:** add per-layer GDN numerics comparator (`5e7a31da`)
- **bench:** surface matched quant diagnostics (`bf473f5f`)
- **bench:** surface Q3 speculation diagnostics (`744ff7c2`)
- **bench:** bind speculation diagnostics to quant comparison (`91161adc`)
- **bench:** retain Qwen AR MTP differential timing (`ba6e4873`)
- **qwen:** emit reviewable quant handoffs (`5952e6e6`)
- **qwen:** gate IU4 construction on Q3 evidence (`bc9906b6`)
- **qwen:** stage matched IU4 hardware lane (`37a65470`)
- **qwen:** seal matched Q3 IU4 benchmarks (`6ae5a2f7`)
- **bench:** attest Qwen speculative timing costs (`e1d24219`)
- **bench:** retain speculative timing breakdown (`ddb931fe`)
- **qwen:** reuse unchanged quant artifacts (`b7d60e29`)
- **ci:** fully benchmark Q3 proof candidate (`b7538136`)
- **qwen:** prove first token on Q3 PLE (`4b76cc5f`)
- **qwen:** add ROCmFPX Q3 PLE recipe (`a06a7ea8`)
- **qwen:** stage pinned quality judge (`9737f6b8`)
- **qwen:** certify sealed vision runtime (`fe2ad51a`)
- **qwen:** seal deployable release artifact set (`cacaa35a`)
- **qwen:** add sealed deployment mode (`4c9d22f7`)
- **qwen:** gate Hugging Face candidate publication (`c0296326`)
- **qwen:** harden measured bakeoff handoffs (`e58581e2`)
- **qwen:** orchestrate measured quant selection (`118e5ab0`)
- **quant:** decouple Qwen MTP selection (`5ef04ee4`)
- **quant:** add serial Qwen candidate bakeoff (`6c617d96`)
- **quant:** plumb Qwen mixed-format bakeoff arms (`ef38beea`)
- **qwen:** enforce matching MTP quant contracts (`255ebe53`)
- **quant:** capture stock Qwen intervention directions (`914a0855`)
- **quant:** add measured Qwen bakeoff controls (`7ac51087`)
- **qwen:** add Flash Next inference foundation (`96ac183b`)
- **server:** accept DeepSeek thinking objects (`c99e35f0`)
- **bench:** measure whether speculation changes a post-tool decision (`d8d419b6`)

### Fixed

- **deepseek4:** write ncall=1, matching the production imatrix convention (`5092b9cd`)
- **deepseek4:** report imatrix coverage, do not refuse to write (`63b9a3b1`)
- **deepseek4:** count chunks in drain, or a good collection refuses to write (`d343b772`)
- **deepseek4:** refuse to write an imatrix with nothing in it (`2323ad99`)
- **deepseek4:** three defects that would have silently corrupted the imatrix (`82bbf2df`)
- **deepseek4:** report monolithic speculation cycles (`44e7f5e2`)
- **deepseek4:** remove vision text-prefill host overhead (`f4bba5a2`)
- **deepseek4:** parse activation dump path in backend init (`c4ffc556`)
- **engine:** bind layer capture to authority bundle (`39d0be15`)
- **engine:** admit bound single-layer control model (`66c35011`)
- **vision:** share router bias tensor suffix (`c0481276`)
- **vision:** share router bias tensor spelling (`03e635a6`)
- **vision:** match converted router bias names (`62171d87`)
- **vision:** reject special artifact files (`88adff8e`)
- **vision:** bind snapshot cache policy to request media (`862c62a9`)
- **engine:** preserve KDA in GDN stride guard (`efbc1cd5`)
- **backend:** guard append-only validation report ABI (`5ace0b90`)
- **validation:** enforce TV gate in release evidence (`6d078e87`)
- **validation:** bound distributional divergence, not just tokens and margins (`4cdf5c38`)
- **engine:** mirror QSA norms in F32 (`8c67086d`)
- **sse:** deliver false-positive tool-marker text instead of dropping it (`4926b93e`)
- **engine:** keep mean reduction tree shape-invariant (`86a5ce1c`)
- **engine:** keep sum_rows tree shape-invariant (`9f1dc333`)
- **engine:** make attention numerics mask cover GDN (`5258cc61`)
- **test:** clear the strict-build red in the rope oracle (`e7a0da30`)
- **server:** correct the Responses image-input error (`a80542d7`)
- **engine:** stabilize zero-row activation quantization (`6ec8125c`)
- **ci:** keep profiler residue out of checkout (`b3b16e33`)
- **bench:** bind decode diagnostics to final selection (`33fb0b8a`)
- **qwen:** execute first-token evidence gate (`38f1fe18`)
- **build:** bind incremental ROCm provenance (`c00c4b7d`)
- **ci:** retain Qwen hardware evidence (`dfa5814c`)
- **qwen:** retain intervention reconstruction input (`82c95eaf`)
- **qwen:** require image-grounded vision evidence (`ab30ac2b`)
- **engine:** align Qwen MTP positions (`15f6a970`)
- **ci:** bind Qwen attestation signer identity (`c81a3324`)
- **ci:** bound Qwen GPU proof phases (`9f01e6ae`)
- **ci:** guard workflow GPU lock handoffs (`a4eacac8`)
- **profiling:** apply calibrated ROCm 10 counter scales (`c5cb7a28`)
- **profiling:** recognize ROCm transaction counters (`4f8efac5`)
- **profiling:** compile counter traffic on ROCm 10 (`9a805b8a`)
- **profiling:** avoid nested calibration GPU lock (`3a7cea61`)
- **server:** preserve assistant refusal history (`4db0d913`)
- **qwen:** align vision width and compatibility docs (`8a0f0260`)
- **ci:** restore checkout-free counter calibration (`198d8d10`)
- **engine:** share Qwen4Exp M-RoPE position walk (`65b91058`)
- **server:** skip vision tool-turn KV snapshots (`dabec0b0`)
- **profiling:** resume completed Qwen passes (`7a4a4273`)
- **profiling:** bound Qwen trace workloads (`97fb3afc`)
- **bench:** calibrate exact Qwen timing shapes (`a5add816`)
- **ci:** authenticate retained Qwen profiler image (`7f779ea3`)
- **ci:** bind retained Qwen profile source path (`a661bb46`)
- **profiling:** resume Qwen ROCm 10 counter passes (`d705812d`)
- **profiling:** bind GPU groups by numeric gid (`77df83ed`)
- **bench:** bind no-eligible Qwen kernel evidence (`ab065149`)
- **engine:** chunk Qwen MTP dense sync rows (`2a984718`)
- **bench:** retain Qwen timing failure details (`bf742cb5`)
- **engine:** confirm Qwen MTP accepts with q1 (`f9613056`)
- **engine:** persist Qwen snapshot frontiers (`95386a9c`)
- **qwen:** make Q3 reuse requests retryable (`aeafe190`)
- **qwen:** allow cache-free candidate reuse (`b381fcaf`)
- **qwen:** materialize GDN norm views for HIP (`d18d3f0b`)
- **qwen:** cast BF16 MTP shared gate for HIP (`355933f2`)
- **qwen:** canonicalize MTP shared gate rank (`6f543a1e`)
- **qwen:** route retained proof diagnosis (`22126268`)
- **qwen:** retain first-token failure diagnostics (`3faa4176`)
- **qwen:** validate selected PLE tensor format (`26fe08f2`)
- **qwen:** grant first-token attestation permissions (`23d4bf1b`)
- **qwen:** normalize repository identity (`b4957bd8`)
- **qwen:** bind sweep extraction identity (`91bc1437`)
- **qwen:** accept current sweep plan schema (`43e7b61b`)
- **qwen:** hash candidate release profile (`70d60fce`)
- **qwen:** preserve compatible activation capture (`dd0ef459`)
- **qwen:** validate emitted cache resources (`cd014a6a`)
- **qwen:** accept bounded cache cleanup evidence (`2a1b96d6`)
- **qwen:** reuse cache across splitter rebuilds (`f96acbe1`)
- **qwen:** derive selection manifest from contract (`835a9944`)
- **qwen:** stage protected selection corpus (`cd402917`)
- **ci:** reclaim obsolete Qwen tooling (`76e42e99`)
- **ci:** resume exact stale image reclaim (`1bf43cfb`)
- **qwen:** normalize cache split prefix (`4b038f69`)
- **qwen:** bound rolling artifact retention (`13e272f1`)
- **qwen:** name sanitized converter subprocess (`68d14a53`)
- **qwen:** expose exact phase handoffs (`3c50b692`)
- **qwen:** name unprivileged converter identity (`126c4ff8`)
- **ci:** keep reclaim runner checkout-free (`d3990664`)
- **ci:** validate dev image provenance by digest (`3e742854`)
- **ci:** preserve construction storage margin (`ed4424ed`)
- **ci:** reclaim final stale image pair (`631a9536`)
- **ci:** avoid privileged legacy tooling cleanup (`2afb7d70`)
- **ci:** retire exact stale Qwen build state (`5551fb2d`)
- **container:** retain verified ROCm version marker (`bc3b0999`)
- **ci:** pin W4A8 evidence upload action (`79e09993`)
- **qwen:** serialize lazy vision provider (`fe9ba5c5`)
- **qwen:** bind exact vision tensor inventory (`04d058eb`)
- **qwen:** hand off snapshot coordination lock (`37e86c20`)
- **qwen:** disable root-owned converter ccache (`4c69d085`)
- **qwen:** hand off retired shard ownership (`09de35a6`)
- **qwen:** map captured stock shard namespace (`7e255b85`)
- **container:** label exact dev revision (`97f5e7f5`)
- **qwen:** initialize retirement workset safely (`0462ab9a`)
- **qwen:** capture residual writer directions (`35eca745`)
- **qwen:** load canonical split expert projections (`3afa5b3d`)
- **qwen:** match canonical MRoPE metadata (`b301ddeb`)
- **qwen:** accept canonical signed metadata arrays (`6bc799ba`)
- **engine:** load canonical split Qwen models (`78261de8`)
- **engine:** harden IU4 ISA evidence gates (`bd374bc8`)
- **qwen:** retain failed model-load diagnostics (`76e9a023`)
- **qwen:** separate capture tool provenance (`c0e6fa27`)
- **qwen:** use numeric GPU device groups (`a55a2709`)
- **qwen:** normalize capture artifact access (`0ef69c4d`)
- **qwen:** stage protected capture corpus (`f2907f7e`)
- **qwen:** bind capture to current bakeoff recipe (`377298dd`)
- **qwen:** validate canonical split metadata (`de521196`)
- **qwen:** canonicalize bounded split prefix (`4af543cc`)
- **ci:** bound Qwen bakeoff expressions (`e8e28688`)
- **qwen:** bound conversion and dispatch handoffs (`2f99a12f`)
- **engine:** preserve ROCMI4 IU4 fragment order (`e0d66232`)
- **ci:** route Qwen candidate build kind (`c180c828`)
- **quant:** harden Qwen candidate encoding (`10236b84`)
- **quant:** preserve and recycle Qwen controls (`ebf39327`)
- **quant:** stream Qwen PLE staging as f32 (`9e44f17e`)
- **ci:** inspect Qwen snapshot lock as root (`d8c64cc5`)
- **quant:** bound Qwen PLE conversion memory (`936ac6f1`)
- **qwen:** bind conversion and companion evidence (`051b4384`)
- **ci:** trust pinned conversion checkouts (`651fa430`)
- **qwen:** bind final gate to shards and live memory (`1df92ac5`)
- **quant:** enforce live gfx1151 GTT ceiling (`a5a1e57c`)
- **quant:** audit ROCmFP4 intervention bytes (`ec2d7a42`)
- **container:** close Qwen vision runtime dependencies (`030d3f3c`)
- **ci:** the benchmark job cannot take the GPU lock either (`a67b57d6`)

### Performance

- **site:** drop two measurement runs that were filed as releases (`dc04216e`)
- **rocmfp2:** affine offset term from block_q8_1.ds.y, opt-in (`912459fd`)
- **rocmfp2:** drop the identity v_perm_b32 from the FP2 unpack (`86d8883c`)
- **rocmfp4:** UE4M3 scale decode through the f16 converter, opt-in (`b307ddf1`)
- **ledger:** record the two ROCm 10 text bundles beside 2026.8.24 (`57ddf518`)
- **deepseek4:** reuse the prefill embed buffer across chunks (`adc8319f`)
- **fattn:** rotate the q rope tail before the D=512 kernels (`f5ad83fb`)
- the engine already has the instrument this sweep was substituting for (`44e52a3a`)
- **deepseek4:** reuse the exact-prefill attention metadata arena (`64ae212f`)
- **deepseek4:** reuse the prefill embed buffer across chunks (`f8bd4563`)
- methodical hot-path sweep; one kept finding, two large ones ruled out (`95530a98`)
- **engine:** keep QSA preparation on device (`1ee72b86`)
- **engine:** batch Qwen graph barriers (`faa5307a`)
- **engine:** quantize strided activation slices in place (`63435cf3`)
- **engine:** mask Qwen batch numerics controls (`a3a50c43`)
- **engine:** isolate Qwen layer-major numerics (`89eaee30`)
- **engine:** report Qwen subsystem magnitude ratios (`39de43ec`)
- **engine:** classify Qwen batch error geometry (`b4c42008`)
- **engine:** isolate Qwen batched subsystem numerics (`f5fe58d8`)
- **engine:** trace Qwen frontier logit margins (`dca7c0e9`)
- **engine:** expose Qwen cross-kernel numerics (`c5612122`)
- **qwen:** avoid retention-only requants (`0708dc53`)
- **qwen:** reuse full differential first token (`01d1b32e`)
- **qwen:** reuse artifact integrity proofs (`81356e2d`)
- **engine:** fuse Qwen final vocabulary head (`f986d1a5`)
- **engine:** persist Qwen MTP QSA graph (`51e693da`)
- **engine:** batch Qwen MTP hidden projection (`3dea183c`)
- **engine:** fuse Qwen HC mixer graphs (`9b1523ef`)
- **engine:** batch Qwen PLE projections (`a8b244eb`)
- **engine:** batch Qwen QSA input projections (`8eef528c`)
- **engine:** batch Qwen verifier output projections (`c51e5239`)
- **engine:** batch Qwen prefill HC projections (`924719fc`)
- **bench:** pin bounded Qwen profiler shapes (`207203ae`)
- **ci:** shorten Qwen first-token prefill (`802397d1`)
- **engine:** cache Qwen YaRN inverse frequencies (`d02cb21d`)
- **profiling:** add ROCm counter unit calibration harness (`cca0463e`)
- **runtime:** cache verified model identities (`9414b44a`)
- **qwen:** accelerate verified benchmark retries (`febb2825`)
- **engine:** bind W4A8 dispatch to gfx1151 ISA (`5aa6f2b6`)
- **engine:** bind IU4 evidence to ISA variants (`bf8bae7d`)
- **engine:** prepack exact IU4 activations (`3fefaeaa`)
- **engine:** add exact gfx1151 W4A8 IU4 experiment (`494f4d28`)
- **engine:** pin gfx1151 IU4 ISA contract (`34f83286`)
- **engine:** batch Qwen GDN prefill (`51ed9466`)
- **engine:** fuse Qwen q1 frontier graphs (`90bf8f9f`)
- **qwen:** batch MTP cache projections (`3f4d1ee7`)
- **qwen:** avoid redundant accepted-token draft heads (`a7a6b890`)
- **qwen:** batch MTP prompt cache synchronization (`c4674ba4`)
- **qwen:** remove QSA and MTP host bottlenecks (`85d3f207`)
- **qwen:** batch ordinary prompt prefill (`bd393558`)
- **qwen:** batch frontier MoE prefill rows (`1afb79ad`)
- **qwen:** fuse frontier MoE and batch MTP verify (`02a7c999`)
- **bench:** measure the released 2026.8.24 image (`6afb2976`)

### Build and CI

- grant actions:read to the container call, and check callers statically (`716c4d6c`)
- install PyYAML where ctest runs, and assert the two CI systems agree (`b42cadab`)
- prove certification from run history, and delete the release PAT (`99eef8d1`)
- audit the path after the gate, not just the point it failed (`71219152`)
- install a python with a standard library, and prove it before the box (`fb60e90d`)
- scope the artifact digests to the job, and check for undefined variables (`56457040`)
- gate certification on the vision behavioural arms (`bb351749`)
- mount the vision artifacts under the names the entrypoint looks for (`86d668f5`)
- take the qwen pipeline off main (`7924d17f`)
- make the actionlint gate pass on GitHub, where shellcheck exists (`0436a62d`)
- grant the certify call the permissions its nested jobs request (`e8a94185`)
- baseline coverage floors for the vision and qwen sources (`17a6c3e7`)
- certify the vision model with its tower, and repair the release-script tests (`b8ee0c67`)
- **qwen:** resume corrected timing from retained evidence (`b8bdf35b`)
- **qwen:** add bounded timing HTTP repro (`2692c799`)
- **qwen:** widen retained differential trace (`ef2f3ef1`)
- **qwen:** retain full differential diagnostics (`c610c5be`)
- **qwen:** derive candidate requests on runner (`3f27f817`)
- **qwen:** plan quality capture on runner (`1b8d3d29`)
- **qwen:** dispatch quality and bakeoff workflows (`ab03eec5`)
- **qwen:** reclaim exact dangling build images (`cbdb15f6`)
- **qwen:** inventory guarded runner storage (`fb831bbc`)
- **qwen:** inspect failed converter residue (`b0962c8e`)
- **qwen:** automate stock activation capture (`27a4ab00`)
- **qwen:** build matching MTP control artifact (`0dd3f9ea`)
- **qwen:** automate bounded control conversion (`3ce8d48f`)
- **qwen:** gate real-weight gfx1151 performance (`e24c82f5`)
- **qwen:** fetch pinned source snapshot resumably (`8ef69081`)
- **qwen:** add read-only gfx1151 preflight (`bf40d075`)

### Documentation

- **vision:** 0v settles the matched-drafter question — ship with the 0731 drafter (`3ccf1e5e`)
- **vision:** the all-zero vl bias is the drafter's three stages, not the LM's 43 layers (`4be15a25`)
- **findings:** FA q-tail prepass measured on gfx1151 and rejected — prefill kernel p50 +33% (`1b6c1b0d`)
- **findings:** MWAITX arm measured and dropped; runtime env search closed (`0e8f2d9c`)
- **findings:** DEBUG_CLR_DIRECT_DOORBELL hangs the measured request; moved to do-not-use (`4f57fd99`)
- **findings:** FP2 transposed layout — reader inventory, repack hook, and the VALU-bound discriminator it waits on (`90f400e8`)
- **vision:** host-side audit of the matched drafter finds no artifact-level defect (`f77c570a`)
- **vision:** ledger the behavioural gates and throughput rows A/B (`404dcb1b`)
- **findings:** rocm-systems runtime review for gfx1151 (`7ba7473c`)
- audit every hardware claim against the ISA; two corrections (`3789cfff`)
- per-layer mask rebuild - blanket hoist is wrong, memoise by ratio instead (`dfb38eb8`)
- ISA review; half the dp4a in the 2-bit expert kernel is redundant (`093808a5`)
- what ROCm 10 actually offers us; system-opt guide already satisfied (`23af1917`)
- **vision:** record native hardware evidence (`326255b0`)
- **vision:** make the C1 base-type comparison executable as written (`f58ae268`)
- rewrite the graft plan around two findings that shrink it (`63713609`)
- scope the vision graft for this fork (`a66a2f7c`)
- **coord:** direct codex to base part B on the ember-vision fork (`f422ed36`)
- **vision:** fix part C release criteria before any measurement exists (`88184624`)
- **coord:** review codex part B plan; disk prefix clamp and worktree base (`dae26f4f`)
- **coord:** vision converter proven, bias_vl landed, branch rebased off Qwen work (`dc1e6e57`)
- **coord:** shelve Qwen and preserve handoff (`5f527082`)
- **coord:** new goal -- requant, support and benchmark the vision model (`8c2d15d3`)
- **coord:** stop Qwen work; new goal is the vision quant (`c57e998d`)
- **coord:** approve the GDN layout-contract regression (`6abdf4a5`)
- **coord:** dispatch depth-4096 cell and shadow diagnostic build (`309f60c3`)
- **coord:** approve the corrected GDN guard; record my assert was wrong (`94638563`)
- **coord:** report corrected GDN stride guard (`5f78ac5f`)
- **coord:** accept the GDN assert defect; green suite could not have tested KDA (`6a648732`)
- **coord:** confirm both bare-AR cells; correct the premature stall call (`9e2f7b27`)
- **coord:** release completed bare AR benchmark (`4b18d4a5`)
- **dead-code:** corroborate sync_fallback on the live artifact (`66f9cbf9`)
- **dead-code:** re-verify the rotation entry against the live artifact (`2b3ad36a`)
- **coord:** mark backlog item 12 done (`9ddf5680`)
- **coord:** record open threads while codex is unresponsive (`5877870b`)
- **perf:** refresh the read-first header for the new target and first measurements (`cbaec432`)
- **coord:** depth-8192 prefill cost warning and cheaper alternatives (`8f33ca42`)
- **perf:** why KV state is host-resident and what moving it would cost (`cea88227`)
- **coord:** propose depth-8192 falsifier within the held claim (`7e53dd02`)
- **perf:** depth-2048 decode 8.14 t/s; the finding is the slope (`75331c05`)
- **perf:** correct the barrier budget -- static sites are not per-token count (`1427c595`)
- **perf:** budget the 0.50x decode gap against the host-barrier census (`8fe21fe7`)
- **perf:** first valid Qwen measurement -- bare AR decode 13.84 t/s at depth 512 (`46e72a3c`)
- **coord:** reclaim deep bare AR cell (`3e37373a`)
- **coord:** withdraw msg 427; empty MMQ inventory is the pass condition (`bf763272`)
- **coord:** release partial bare AR attempt (`a7ff6ab1`)
- **coord:** depth-512 dispatch proof is vacuous (`60ae45a5`)
- **coord:** dispatch proof passes on an empty inventory (`d08b99e3`)
- **perf:** correct the projection-compare provenance (`78712b33`)
- **perf:** the startup self-check already agreed at the failing widths (`731be831`)
- **coord:** reclaim corrected bare AR benchmark (`71f1664b`)
- **coord:** correct the benchmark void rule to the measured window (`9a5e3120`)
- **coord:** close void benchmark task (`948b609a`)
- **coord:** void exact-prefill bare AR attempt (`5b05e9ef`)
- record the withdrawn pool-tail hypothesis; advance waterline to 4885363 (`b792d10d`)
- **coord:** withdraw the pool-tail hypothesis (`48853639`)
- **coord:** claim exact-prefill bare AR benchmark (`a3864fb8`)
- **coord:** approve exact-prefill benchmark flag (`db972770`)
- **coord:** request exact-prefill flag review (`0f8b3cd9`)
- **perf:** close isolated quant oracle program (`cf66c37b`)
- **coord:** pool-tail hypothesis for the composition defect (`a4331bdd`)
- **coord:** take exact-prefill benchmark work order (`108c138f`)
- **coord:** authorise and spec the bare AR decode benchmark (`a881f479`)
- **coord:** shadow diagnostic needs a green-width noise floor (`117cd561`)
- **coord:** start full-graph shadow discriminator (`453fdc24`)
- **coord:** approve live-type oracle ledger closure (`cbeed0a0`)
- **coord:** request live-type oracle ledger review (`425655a8`)
- **coord:** invoke the stopping rule; isolated axes exhausted (`278899a8`)
- **coord:** release live-type oracle runner (`57ffda1c`)
- **coord:** advance waterline to 73f4ce1; record the artifact-identity rule (`d1106749`)
- **coord:** retry live-type oracle preflight (`73f4ce11`)
- **perf:** size the GDN transpose-then-concat as the first prefill lead (`3405c371`)
- **coord:** claim runner for live-type oracle (`a9defee1`)
- **coord:** approve dense/routed width split (`33d343f8`)
- **coord:** correct routed oracle width (`8cb5309b`)
- **coord:** approve type-101 operator oracle (`e7ecc4eb`)
- **coord:** strengthen oracle review evidence (`05b359bb`)
- **coord:** request ROCmFP4 oracle review (`5507da13`)
- **perf:** record activation inventory result (`040dd484`)
- **coord:** approve revised inventory ledger entry (`e383aae6`)
- **coord:** acknowledge corrected performance bar (`fcc5a1aa`)
- **goal:** correct the bar to the agentionai Qwen3.8-Flash-Next card (`5eef5f38`)
- **coord:** notify goal change (`a6c411ad`)
- **goal:** raise the performance bar to exceed the Strix Halo Vulkan fork (`bf59cd6f`)
- **coord:** correct inventory oracle conclusion (`d69d1184`)
- **coord:** reject stopping conclusion; type-101 arithmetic untested (`1fe79e34`)
- **coord:** request activation inventory ledger review (`d805ffd1`)
- **coord:** release runner after corrected inventory (`5391a99b`)
- **coord:** correct inventory target and reclaim runner (`fb8edc21`)
- **reference:** review the LaurentZuijdwijk Strix Halo llama.cpp fork (`127e58f1`)
- **coord:** release runner after activation inventory (`e1fe9d9c`)
- **coord:** pre-register inventory interpretation (`000feca4`)
- **coord:** report inventory staging retry (`b85bfdcc`)
- **coord:** advance waterline to 9370fc4; record the structural-claims rule (`3831e214`)
- **coord:** claim runner for activation inventory (`71930a38`)
- **coord:** close validation ABI guard (`9370fc4e`)
- **coord:** approve full-member ABI guard (`9ada1202`)
- **coord:** ABI guard gap proved empirically (`a6ecad60`)
- **perf:** record adopted prefill TV gate (`65cf7b36`)
- **coord:** withdraw ABI append claim; codex was right (`f7b040bc`)
- **coord:** approve ledger correction (`fe56a904`)
- **coord:** approve TV integration and inventory deltas (`b0c17366`)
- **perf:** record ROCMI4 row-tail oracle (`c718ef27`)
- **coord:** approve row-tail ledger; grouped-src path is DeepSeek-only (`64620406`)
- **perf:** propose total-variation distance at serving temperature as the criterion (`3051127a`)
- **coord:** approve write-extent guard with failure-reporting requirement (`cb7c7187`)
- **coord:** approve telemetry refactor (`f5b77ecf`)
- **coord:** src1 inventory and write-extent sentinel (`31dbba2a`)
- **coord:** dispatch proof approved; W4A8 config question (`7e49d906`)
- **coord:** row-tail oracle review (`52afbe07`)
- **perf:** record partial-K oracle falsification (`de551fd5`)
- **coord:** approve codex ledger delta; N values for the next sweep (`a4648b1d`)
- **perf:** predicted defect -- MMQ ROCMI4 tile loader has no K bound (`47df7817`)
- **coord:** loader has no K bound; mechanism corrected (`2a3e5e57`)
- **coord:** approve partial-K oracle with exactness guard (`d5a8bcb3`)
- **coord:** green validator with broken logits (`d8fcf2be`)
- **perf:** validator-green run with r=0.56 logit correlation (`0c0e04b6`)
- **perf:** codex's oracle and dense-crossover ledger sections (`122d1d66`)
- **coord:** partial-K verified from source arithmetic (`46323fca`)
- **coord:** MMQ partial-K lead (`246521d4`)
- **coord:** concede bucket hypothesis; dense crossover confirmed (`80f13a20`)
- **perf:** dense crossover alone is sufficient; bucket refuted as necessary (`41016deb`)
- **coord:** width 17 correction and baseline check (`446cca2c`)
- **perf:** width 17 chunks to 16+1, not bucket 0 (`0ebf6994`)
- **perf:** arms 2 and 3 are collinear; routed ceiling is 8 for ROCMI4 (`7573f4d0`)
- **perf:** confound has a third arm; NCOLS moves only the dense boundary (`2a3f4e73`)
- **perf:** record the width-6 confound between matmul family and MoE bucket (`02ba3cb3`)
- **coord:** approve inexact quantizer case (`00b45beb`)
- **coord:** operator oracle review (`52e990d9`)
- **perf:** q1-vs-production correlation collapses at widths 6/17 (`e2c2df69`)
- **coord:** cross-evaluation handoff (`d3ec7af5`)
- **perf:** cross-evaluation of divergent argmax rows (`fd5f665e`)
- **coord:** ledger review; own the git add -A error (`c21b01f9`)
- **perf:** rank-aware analysis of the F32 reference and the default path (`08ffdad0`)
- **perf:** F32 reference width-2 gate red; criterion statistic is rank-blind (`137f8b60`)
- **coord:** correct scope overreach; restore-check question stands (`e3651027`)
- **coord:** production is up, preflight unit name wrong (`839355a7`)
- **coord:** F32 delta builds confirmed (`e9f11e42`)
- **perf:** restore the sync_fallback denominator to the ledger (`e81d8bea`)
- **coord:** F32 comparison tool handoff (`c51d516e`)
- **dead-code:** sync_fallback is live in the F32 reference build (`2c4bea82`)
- **dead-code:** narrow the GGML_PREC_F32 entry to flash attention (`4215387f`)
- **coord:** F32 reference mechanism review (`c21d6b4e`)
- **coord:** tranche 1 measurements and bit-identity correction (`25a906e2`)
- **perf:** tranche 1 hardware correctness at 8c67086 (`eaca8971`)
- **coord:** advance waterline to 8c67086 (`915b4045`)
- **coord:** withdraw realloc hazard, unreachable (`c1826365`)
- **coord:** mirror lifetime review green (`094dd935`)
- **coord:** BF16 norm widening review green (`83e1a3e8`)
- **coord:** tranche 1 MTP assert root cause (`128dd33c`)
- **coord:** F32 reference collection spec (`c6148382`)
- **coord:** record independent full-suite verification (`037d7433`)
- **perf:** re-derive host-barrier census at 1ee72b8 (`7f549218`)
- **coord:** record batch-shape arithmetic sweep (`6d16eab5`)
- **coord:** criterion hole analysis (`69360238`)
- **coord:** advance waterline to 1ee72b8; record independent-build rule (`812ceee4`)
- **coord:** resolve INPUT flag question from source (`8c00a59e`)
- **coord:** review tranche 1 (`3302f425`)
- **coord:** the F32 reference needs no new mechanism (`be21bca8`)
- **coord:** work order to convert and load the MTP companion (`760cac74`)
- **coord:** work order for the F32 dequantized reference at width 6 (`8bde7d32`)
- **coord:** the green-width floor is zero by construction and cannot calibrate (`ec6859aa`)
- **coord:** verify both landed commits, advance waterline to 5ac6d95 (`134cc49e`)
- **coord:** review the exact-stream noise-floor metrics (`0db265a5`)
- **perf:** the decided criterion would have accepted the sum_rows defect (`eeea24bb`)
- **coord:** review the Q4_K MUL_MAT_ID test (`514c1621`)
- **coord:** record the batch-shape arithmetic lesson where agents read it (`88458a4e`)
- **coord:** review the margin criterion, tighten my own spec (`175c8e45`)
- **coord:** codex halts between turns because steer needs an active turn (`b3c22ab2`)
- **coord:** work order for the Q4_K allow-list and MUL_MAT_ID check (`59d87a39`)
- verify Q4_K needs no backend work, only the allow-list (`164e55e3`)
- **perf:** correct the claim that nothing published exceeds our gates (`c81bf20a`)
- assess three third-party Qwen GGUFs without downloading them (`c40a20b3`)
- **coord:** hand the decided criterion to codex (`37380067`)
- **perf:** record the user's release-criterion decision (`b7f9c0f2`)
- **perf:** put the current state at the top of the ledger (`e25e8a62`)
- **coord:** tranche 3 mapped from the reference, and ordered last (`40fe9d89`)
- **perf:** width 17 completes the correlation rather than complicating it (`af34ffc3`)
- **perf:** the width boundary is exact and the remaining question is the criterion (`d59ee4eb`)
- **perf:** the reference has no q1 path, so the blocker is self-imposed (`a3f379be`)
- survey other engines on this silicon; one portable candidate (`349b90a5`)
- **perf:** stop asserting a CU count I had not verified (`abab0563`)
- **coord:** hold the composition diagnostic, the mask data is the family signature (`38938de3`)
- **perf:** widths 2 and 3 green, width 6 is a kernel-family seam (`b5ab930e`)
- **engine:** register latent mean reduction path (`16db4a73`)
- **coord:** tranche 1 mapping, and the reference picks Path 2 (`014f22ff`)
- **coord:** approve the mean dead-path entry with a criterion guard (`ccb68eb1`)
- **coord:** verify the mean twin and the amend recovery (`5055e5cf`)
- **perf:** record the mean.cu twin, the guard, and the path to a publishable number (`a6ef37e9`)
- **coord:** file the sum_rows invariance guard (`706313aa`)
- **coord:** mean.cu carries the same reduction-shape defect (`9979841a`)
- **coord:** approve the sum_rows fix, flag the shared DeepSeek path (`f40dd557`)
- **perf:** root cause -- sum_rows picks its reduction tree from the row count (`f9ed55f2`)
- **coord:** file the tranche 2 spec derived from the reference (`85e8fbc8`)
- **coord:** approve the GDN input capture after the view-liveness fix (`8afba5ff`)
- **coord:** approve the HIP test switch, restate the view blocker (`75955fbc`)
- **coord:** request changes on the GDN input capture (`194578e3`)
- **perf:** unroll2 cleared, and the divergence is exactly one float32 ULP (`4536878a`)
- **perf:** raise the bit-exactness question as a user decision (`ea03e5e9`)
- **perf:** record the 345-prefill reference implementation (`3748fc8a`)
- **perf:** correct a false scope claim I introduced about codex 338 (`fc3fba73`)
- **coord:** advance the waterline over codex's two engine commits (`1377ad77`)
- **perf:** the S_v=128 kernel passes on HIP at n=3 with non-zero state (`4bf36b4e`)
- **coord:** file the seam eliminations and the g/beta stride coupling (`ddbd636c`)
- **coord:** approve codex's GDN batch comparator (`d5af0df0`)
- **perf:** grouped-cols exonerated, and the synthetic control was vacuous (`3e2047c4`)
- **perf:** narrow the GDN defect from the pass/fail pattern (`70d68c60`)
- **coord:** record grok's budget exhaustion and reassign its open work (`cf0cc19b`)
- **perf:** record the grouped-cols selection conditions from source (`41c27520`)
- **coord:** document the push channels and how each one breaks (`ebe960db`)
- **perf:** the blocker is isolated to run_gdn_batch, and my MMVQ lead is withdrawn (`b84f8b24`)
- **coord:** file the index-1 question and the channel fix note (`da2bfc54`)
- **coord:** locate the tool_call silent truncation precisely (`0d2e244a`)
- **coord:** scope the GDN control question (`b329b2a1`)
- **perf:** record the faa5307 hard gate and the MMVQ specialization suspect (`fbfdbfe4`)
- **coord:** file the strict-green confirmation and the bit-4 review (`eec1c68c`)
- **coord:** approval queue directories (`7c40b54b`)
- **coord:** add the missing codex backlog (`db42211c`)
- **perf:** record that the GDN fixture is a shape HIP refuses (`3d2c7ade`)
- **coord:** file the GDN coverage caveat (`cdfb9604`)
- **perf:** qualify the GDN coverage row and the green-branch discriminator (`ec6d97a8`)
- **coord:** broadcast the LOOP.md staleness fix (`5560086b`)
- **coord:** stop LOOP.md carrying its own copy of the measurements (`cc2d3d99`)
- **perf:** coverage of batched subsystems is complete at the failing width (`cda41a6d`)
- **perf:** record eight eliminations from the mask-31 fail branch (`faa7cbac`)
- **coord:** record that both composition asymmetries are inert (`2fa14c6c`)
- **coord:** file the mrope history asymmetry as a deferred review finding (`b9cc8106`)
- **perf:** name the one run that separates the remaining hypotheses (`cb554a26`)
- **perf:** record the MoE pad elimination and rank what remains (`5da269a4`)
- **perf:** record the dense pad-independence elimination (`9326d7ab`)
- **coord:** file the not-precision finding (`4d57e81b`)
- **perf:** withdraw the "root cause isolated" claim on the correctness blocker (`7b33b5f7`)
- **perf:** qualify the "root cause isolated" claim on the correctness blocker (`6ea02a18`)
- **coord:** file the dense-boundary finding (`47780f1e`)
- register the QSA block scorer as inactive below the 2048-token boundary (`086edc20`)
- **coord:** file the RMS oracle result and the tranche-1 accounting rule (`35117d82`)
- register HIP graph replay, and correct the tranche 1 payoff again (`1d3804d1`)
- **coord:** broadcast the dead-code register rule (`da94a3ae`)
- close the rotation falsifier at the GGUF header (`177707f6`)
- make dead-code tagging a standing rule and fix the live barrier count (`c6d227c5`)
- register engine code that cannot run on the shipped configuration (`c1989483`)
- **perf:** add the barrier census and the rope precision result (`c1a39ef7`)
- **coord:** file the barrier census and withdraw the async A/B prediction (`5d1b7fa5`)
- **coord:** file the rope precision result (`dff98327`)
- **coord:** record the rope oracle result and withdraw the copy attribution (`6eafdbb2`)
- **qwen:** record diagnostic timing and external calibration (`8cdfe465`)
- **qwen:** record the measured performance status (`19ff4d3f`)
- **engine:** clarify RDNA3.5 ISA provenance (`a94a0e3a`)
- **perf:** refresh frontier kernel applicability (`1f913a60`)
- **quant:** record external control fit rejection (`f12d82a7`)
- **quant:** document bounded activation capture (`d667263f`)
- **quant:** clear fixed ROCmFP4 intervention blocker (`bb3de0db`)

### Testing

- cover the paths the coverage gate was failing on (`ee39d363`)
- **vision:** size the over-count offsets array to the count being asked (`51e7670d`)
- **deepseek4:** say plainly that the graph test does not cover the view/base defect (`636735d8`)
- **deepseek4:** exercise the imatrix collector against a real ggml graph (`c1dedd8a`)
- **engine:** pin affine ROCmFP2 layout (`39cf6e93`)
- **vision:** preserve router bias decoys (`511eca0f`)
- **vision:** gate converted router bias inventory (`ca54fb23`)
- **vision:** pin learned marker row endpoints (`e4fcbfc4`)
- **vision:** pin learned marker prefixes (`98347a30`)
- **vision:** probe real learned markers (`1b680311`)
- **engine:** exercise GDN scalar and KDA layouts (`a75fc8ea`)
- **engine:** cover live ROCmFP4-fast arithmetic (`b4e55d01`)
- **engine:** cover ROCMI4 output row tails (`5bfc1353`)
- **engine:** cover ROCMI4 partial K tiles (`b4fb6fe2`)
- **engine:** add ROCMI4 operator oracle (`5b8e3683`)
- **engine:** add F32 dequantized reference capture (`8815442f`)
- **qwen:** guard the reduction-shape invariant that the blocker violated (`f0213097`)
- **qwen:** measure what a token-boundary rounding costs the GDN recurrence (`65131b3a`)
- **qwen:** add a double-precision control for the GDN recurrent state (`18e1253d`)
- **qwen:** put the GDN control on the S_v=128 kernel production actually runs (`4e9a6aa0`)
- **qwen:** give GDN a batch control at a HIP-legal channel count (`56dfb0fa`)
- **qwen:** close the MoE half of the pad-independence claim (`b5d0bb58`)
- **qwen:** make pad independence a test instead of a comment (`99dcc3d6`)
- **qwen:** cover the RMS half of tranche 1 on the strided query view (`4e972da8`)
- **qwen:** measure the rope oracle against an exact double reference (`3cc509e8`)
- **qwen:** oracle the graph RoPE path against the scalar reference (`a7c79be0`)
- **engine:** compare Qwen production prefill with q1 (`4b7213c3`)
- **engine:** pin W4A4 grid arithmetic (`1532d512`)
- **qwen:** require balanced finalist confirmation (`bd146dd0`)
- **engine:** bind IU4 gate to RDNA3.5 ISA (`bf8eaf7c`)
- **engine:** verify IU4 accumulator dataflow (`960371df`)
- **engine:** harden gfx1151 IU4 ISA gate (`1ae2dc71`)

### Maintenance

- **build:** ignore local build variants (`bbdaee11`)
- **coord:** remove the in-repo .coord copy (`c921646f`)
- **coord:** remove the in-repo .coord copy (`96599b13`)
- **coord:** move coordination state to ~/Projects/.coord (`ddd2d02c`)
- **coord:** cite the AMD machine-readable ISA as the ISA authority (`2c1f5362`)
- **coord:** add the four-agent coordination channel (`e2cafd01`)

### Other

- take the unfinished second-architecture work off main (`3ab9d15c`)
- README: document vision support (`562b93c3`)
- write the release entry for the vision engine (`30045e7a`)
- re-pin after the card link fix (`2cdec52d`)
- re-pin after the card rewrite (`05a40c2a`)
- re-pin after the card edit (`de78c62b`)
- pin the published vision model revision (`d08ee660`)
- default DeepSeek deployment to the vision model and its tower (`b5837402`)
- Merge remote-tracking branch 'origin/feat/ds4-activation-dump' into claude/release-integration (`6a77497d`)
- Merge remote-tracking branch 'origin/perf/ds4-prefill-embed-reuse' into claude/release-integration (`28ac241a`)
- draft the unreleased entry from the branch's own commits (`ebf9cf08`)
- register the real-graph imatrix test under the engine guard (`955cba2e`)
- AGENTS: cite the source for every hardware claim (`fa2698f5`)
- **isa:** VOPD answer was wrong by omission (`9c79668e`)
- refuse a bundle whose spec_cycles counter is inert (`d9061e56`)
- **isa:** expose the whole spec, not just DPP16 and VOPD (`fd20425b`)
- let the site builder preserve published releases (`25eff495`)
- compare on coverage, not equality (`28a0d552`)
- release-over-release delta tool (`f785af0b`)
- let the site builder preserve published releases (`ed083900`)
- compare on coverage, not equality (`de847a54`)
- release-over-release delta tool (`1c4c0d0b`)
- distinguish asserted model digests (`6575096f`)
- require speculative workload evidence (`f8efd7a2`)
- Clarify host GPU group binding (`6d817c78`)
- Bind benchmark containers to host GPU groups (`efe3891c`)
- Bind vision release performance bundles (`ed6cf9ff`)
- add vision workloads to the release-over-release perf harness (`7174441f`)
- **vision:** land the behavioural gate and build revision guard (`447be6cb`)
- **vision:** land the native vision engine from the ember-vision fork (`ca6fb528`)
- Replace the tower-depth numeric gate with a two-armed behavioural one (`830d96de`)
- Key the prefix cache on image content, not just token IDs (`363b6a41`)
- add loaded() accessor for the server-layer encode path (`c4a55510`)
- validation path for image embeddings (ds4-vision stage 2b) (`638d71be`)
- **gdn:** assert g and beta share strides (`1f8ba731`)
- **qwen:** add --ranks, the rank-aware view of a logit disagreement (`0aadac33`)
- **qwen:** offline comparison for the F32 dequantized reference (`3e145d38`)
## 2026.8.24

### Added

- **perf:** show decode and prefill history together, not behind a toggle (`9c654963`)
- **perf:** show the number when one release is selected (`33f01641`)
- **perf:** group workload decode into columns, one per release (`e093985c`)
- **perf:** plot release history as a line, not a row of bars (`8ae85095`)
- **bench:** automate the per-release performance bundle (`ac74130d`)

### Fixed

- **ci:** take the GPU lock through a wrapper, and repair the restore guard (`11954c59`)
- **ci:** block production from restarting during certification (`02a520fc`)
- **ci:** fail loudly when production restarts during certification (`bb0f0824`)
- **ci:** hold the documented GPU lock through certification (`f6815452`)
- **ci:** wait for memory before each certification model load (`a20b8104`)
- **perf:** repair the page, and test that it actually renders (`6460b35b`)
- **ci:** check out the harness in the job that runs it (`885200f0`)
- **perf:** stop peak callouts landing on each other and on the lines (`b6eb39d2`)
- **perf:** crop the release-history axis to the range the data occupies (`994c6b8e`)
- **perf:** make the performance page usable on a phone (`789d4eb5`)
- **ci:** stop TARGET_SHA meaning two things in the benchmark step (`27ec72d8`)
- **bench:** benchmark an old release as it shipped, and keep failure reasons (`212bf2a7`)
- **bench:** plot the depth that was asked for, not the one that drifted (`dba6c388`)
- **bench:** repair four defects the first dry run exposed (`6c97629b`)
- **deepseek4:** abandon speculation after one unqualified profitability pause (`d616074c`)
- **deepseek4:** bound the warmup exclusion so the profit scheduler can still see (`d8993edd`)
- **deepseek4:** correct the warmup bail-out's false monotonicity claim (`b3720a50`)

### Performance

- **bench:** re-measure 2026.8.10, .22 and .23 against the current model (`f8805583`)
- **deepseek4:** qualify the wide verify path on a rate, not a lucky streak (`b66595d6`)
- **ggml:** size the rope block to the row instead of a fixed 256 (`57f11cdf`)
- **ggml:** collapse the short-row reduction to its exact minimal form (`f3da20c2`)
- **ggml:** stop launching one block per row in reduce_rows_f32 (`00556dd1`)
- **deepseek4:** hand back to AR when batch-verify warmup is unreachable (`b2428029`)

### Build and CI

- benchmark the released image in its own job after promotion (`87c2e099`)
- do not certify a push that only changed documentation (`140bbc24`)
- **gfx1151:** benchmark the certified image as part of certification (`adf64ec2`)

### Documentation

- **operations:** clarify storage and release validation (`fa83e82f`)
- refresh contributor and performance guidance (`a9d51367`)
- refresh release and performance guidance (`07ef572c`)
- refresh runtime and performance references (`b4d6348c`)
- link current performance bundle (`40b6e065`)
- refresh release performance summary (`00cb4148`)
- **readme:** name the performance page instead of printing its URL (`865a21cf`)
- **readme:** standard depths in the performance table, and a link to the page (`e060f4e3`)
- **readme:** drop the context-scaling chart from the performance section (`170c772b`)
- **perf:** a browsable performance page, built from benchmark bundles (`f9dfa520`)
- refresh performance numbers from a sweep on the current build (`d24b6d5d`)
- **assets:** give the context chart a legend and peak callouts (`601ce8cd`)
- **readme:** headline plus a measured performance section (`f02650ed`)
- **readme:** put measured performance above the fold (`357c84c8`)
- **perf:** correct the decode and roofline claims, record the prefill work (`2d90f648`)

### Maintenance

- **release:** v2026.8.24 (`721e6389`)

### Other

- hand-written release metadata the automation owns (`0be0764f`)
- **ci:** take benchmarking back out of the certification job (`89bc6dc2`)
- **perf:** dress the performance page in the otheru palette (`487df24b`)
## 2026.8.23

### Fixed

- **test:** assert the compose environment syntax the file actually uses (`5f6241cc`)
- **deepseek4:** stop charging warmup cost to steady-state speculation (`af16fe4b`)
- **docker:** drop compose declarations that only restated engine defaults (`7810971a`)
- **docker:** stop the entrypoint overriding compose, make compose authoritative (`8c999fee`)
- **engine:** correct two profiling-harness bugs found on hardware (`3da625c0`)

### Performance

- **deepseek4:** 33.60 -> 37.49 tok/s decode via width-6 verify (`027ed106`)
- **engine:** extend DPP wave reductions, verified against the RDNA 3.5 ISA (`3a54084c`)
- **engine:** DPP wave reduction in hand-written assembly for gfx1151 (`5f460a0c`)
- **engine:** select WGP vs CU mode per translation unit on gfx1151 (`410d61d1`)
- **engine:** branch-free UE4M3 scale decode for gfx1151 (`b6291ce6`)

### Build and CI

- **docker:** add Hyperloom to the dev image (`88898ac8`)

### Documentation

- **perf:** document TTFT, and record why narrowing the FA KV read failed (`1cb30f76`)
- **engine:** link the two competing LUCE_MMVQ_MAX_NCOLS defaults (`4a1ccfef`)
- **engine:** correct the HIP-graph diagnosis, which blamed the wrong tensor (`dcc00326`)

### Maintenance

- **release:** add gfx1151 kernel profiling harness (`e5a0d064`)
- **release:** document CI gates and clear prune residue (`f02f7d10`)
- **release:** document Forgejo sanitizer split (`b760adb8`)
- **release:** stabilize Forgejo queue (`02305c07`)
- **release:** recover release-note publication (`08f4f9f8`)
- **release:** repair certified tag publication (`45ac7d3c`)
## 2026.8.22

### Curated notes

- Added `--host` and `EMBER_HOST` so trusted container/Kubernetes gateways can
  reach Ember while preserving the unauthenticated loopback default.
- Added `EMBER_VERIFY_EXISTING_SHA256=0` for trusted immutable model stores;
  downloaded artifacts remain pinned and checksum-verified before promotion.
- Integrated and documented the opt-in CPU/GPU/XDNA2 resident DSpark
  prototype. The XDNA Compose overlay now selects the measured two-session
  proposal pipeline instead of the slower target-expert placement experiment.
- Kept heterogeneous inference experimental: the fixed throughput fixture
  passed, while capture-graph output equivalence remains a promotion blocker.
- Linked README installation guidance to the current GitHub release instead of
  embedding a version that becomes stale after every release.
- Made candidate certification and release promotion fully unattended: trusted
  `main` commits proceed from immutable GHCR images through the dedicated
  gfx1151 runner to changelog generation, tagging, and package publication.
- Pinned the certification model pair to the published Hugging Face artifacts,
  added IOMMU/device preflight checks, and made the hardware job quiesce and
  restore production around exclusive GPU validation.
- Upgraded GitHub checkout steps to the pinned Node 24 release so routine CI
  and release logs no longer carry Node 20 deprecation warnings.
- Added persistent BuildKit, ccache, and Trivy database reuse across GitHub and
  Forgejo runners, including compiler cache mounts inside the ROCm Docker build.
- Kept periodic full model digest verification while caching unchanged results
  for seven days and using direct I/O on cache misses, preventing the 96 GiB
  checksum pass from consuming UMA and disabling monolithic DSpark validation.

### Added

- **release:** automate gfx1151 certification (`62205ed3`)
- **release:** automate certified promotion (`324956c7`)
- **ci:** automate candidate release chain (`a44a8205`)
- **ci:** mirror GitHub issues to Forgejo (`ad46d678`)
- **release:** finish deployment controls and XDNA docs (`4c37043a`)
- **xdna:** validate asynchronous DSpark provider (`1b439f8f`)
- **xdna:** schedule weighted expert runlists (`666c612d`)
- **xdna:** add async DSpark draft provider seam (`ac541642`)
- **xdna:** package ROCMFP2 XRT prototype (`e6f98a8d`)
- **engine:** prototype XDNA2 expert offload (`d41d2642`)

### Fixed

- **release:** compare sorted metadata paths (`fff01830`)
- **ci:** mount Forgejo CA in job containers (`4f64a037`)
- **release:** quiesce production supervisor (`27a9effb`)
- **ci:** make IOMMU preflight explicit (`3803c075`)
- **release:** avoid stale README version (`9b918f43`)
- **ci:** isolate release note permissions (`39de69f2`)
- **ci:** use native Forgejo push mirror (`d0ddd613`)
- **engine:** make XDNA corpus target-exact (`7472ac92`)
- **engine:** make resident XDNA acceptance target-exact (`6d0cb138`)
- **engine:** classify custom clamp types (`83a7b839`)
- **engine:** isolate resident graphs and XDNA context (`115d3a79`)
- **xdna:** validate mmap-backed model path (`907d0064`)
- **ci:** let the vulnerability scanner reach its database (`0e57f256`)
- **docker:** declare the image source so GHCR links the package (`0c33af6b`)
- **ci:** give the BuildKit container a working network (`3df0939a`)
- **release:** publish the container image where the docs promise it (`3ab6fada`)

### Performance

- **release:** cache model verification and expose hits (`a629fb25`)
- **release:** cache builds and preserve UMA capacity (`f2627063`)
- **engine:** batch resident verification safely (`06d15d7b`)
- **xdna:** attribute resident pipeline phases (`5a3171c1`)
- **xdna:** gate resident promotion by confidence (`d7d604b0`)
- **xdna:** capture current design control (`56c25ad2`)
- **xdna:** pipeline resident draft proposals (`6f8ba53c`)
- **xdna:** vectorize draft routed experts on CPU (`0c98d313`)
- **xdna:** keep draft Q8 stages resident (`cdec0696`)
- **engine:** fuse DSpark context KV on GPU (`4cbba655`)
- **xdna:** unify draft projection overlay (`358e1794`)
- **xdna:** keep draft projections on fixed overlay (`732be598`)
- **xdna:** parallelize DSpark CPU reductions (`3b830cab`)
- **engine:** measure GPU DSpark preprojection (`54270615`)
- **engine:** stage DSpark main projection on GPU (`f9f9d141`)
- **xdna:** reuse weights across draft context (`4fc7f551`)
- **xdna:** measure task-blocked Q8 projections (`a449b3ec`)
- **xdna:** add compensated Q8 draft shared expert (`efad17c8`)
- **xdna:** mask routed rows in gen8 (`5c97cd24`)
- **xdna:** add five-row ROCMFP4 expert kernel (`6d3b7acb`)
- **engine:** measure fused verifier MoE budget (`e888ef1a`)
- **xdna:** batch shared experts in gen6 (`de55413a`)
- **xdna:** measure concurrent GPU load (`f6e91bd7`)
- **xdna:** validate zero-copy GPU buffers (`76b9d3cf`)
- **xdna:** batch gen5 draft blocks (`8729f21b`)
- **xdna:** fuse expert pipeline in gen5 (`252b813f`)
- **xdna:** expose provider phase timings (`aaa0cf49`)
- **xdna:** vectorize ROCMFP2 decode in gen4 (`e811dfde`)
- **xdna:** preserve projection outputs in fp32 (`2de48a4f`)
- **xdna:** prototype second-generation expert kernel (`4d3c55a6`)

### Changed

- **engine:** prune unsupported backends and prototypes (`eade92dd`)

### Build and CI

- **xdna:** refresh runtime for queued drafts (`452c2e7d`)

### Documentation

- **xdna:** pin gen43 benchmark candidate (`8d0c90fd`)
- **xdna:** record whole-draft contention results (`8a05387a`)
- **xdna:** record AIE-RT queue rejection (`571d5d8c`)
- **xdna:** record gen5 validation scope (`b50e00bb`)
- **ci:** record the self-hosted builder and its fork-PR constraint (`8b9c5f99`)

### Testing

- **json:** cover UTF-8 output boundary (`48e0ca53`)
- **xdna:** enforce resident speculation gate (`37f7f6bf`)
- **xdna:** reject bfp16 draft projections (`9a8b0647`)
- **xdna:** validate trained draft experts (`41454f7f`)

### Maintenance

- **ci:** move checkout actions to Node 24 (`7987558a`)
## 2026.8.10

- Prepared the first source release.
- Added a one-command Docker Compose workflow with resumable model download.
- Pinned the default model revision and added expected-size, free-space, and
  SHA-256 checks before startup.
- Added host/container preflight diagnostics, Compose health reporting, and
  read-only plus generation smoke tests.
- Added OCI image metadata and operations/upstream-release guidance.
- Split the container into a full-toolchain `dev` target and a minimal
  dependency-closure `release` target used by Compose.
- Added licensing, vendor provenance, security, support, and contribution policy.
- Synced the latest agent-progress work: repeated-call diagnostics, a
  result-based progress lease, empty/degenerate-turn telemetry, and detection
  of tool markup delivered as visible text.
- Added opt-in stateless automatic loop recovery and Compose environment
  controls for progress reporting and recovery thresholds.
- Hardened request parsing against embedded NULs, duplicate or mistyped fields,
  unsupported HTTP transfer framing, and excess headers.
- Made strict tool-schema number equality exact beyond IEEE-754's integer range
  and replaced direct POSIX interpretation of ECMA-262 regex patterns with a
  translated, fail-closed subset.
- Removed one-off development probes and made static analysis, exact-commit
  tests, and gfx1151 certification blocking image-release gates.
- Promoted the gfx1151 HIP GEMM batch sweep and recorded the successful
  two-session hardware acceptance result.
- Changed the default Compose path to pull an immutable GHCR image while
  retaining an explicit local source-build override.
- Added project and third-party license notices to the runtime image and made
  the vulnerability-reporting fallback actionable.
- Added GitHub-hosted CPU and large-disk container workflows, direct GHCR
  publishing, and a manual environment-gated gfx1151 certification workflow.
- Fixed the ROCm 7.14 runtime image assembly to package both the gfx1151
  rocBLAS KPACK and its architecture-specific Tensile metadata/code-object
  tree from the pinned toolchain.
- Normalized loader-reported `/lib` paths into merged `/usr` locations so the
  collected runtime can be copied into the Ubuntu 24.04 release stage.
- Hardened gfx1151 certification to verify the immutable model pair,
  separately require exercised DSpark and disk-cache paths, use locked memory,
  and verify live speculative decoding outside resident batching.
- Repointed first-run acquisition to the current published quant and DSpark
  drafter, made both artifacts mandatory and digest-locked, and removed support
  for substituting unverified model files.
