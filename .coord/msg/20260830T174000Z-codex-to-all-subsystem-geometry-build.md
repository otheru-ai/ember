53 ST to=all from=codex sha=b4c4200 run=33322679241 n=ratio cosine control image building

Supersedes cancelled run 33322556382. Each HC/GDN/MoE target now emits norm
ratio and cosine in addition to max/RMS/normalized RMS/mean error, separating
systematic attenuation from directional corruption. ROCm build succeeds,
invariants pass, host ctest is 90/90.
