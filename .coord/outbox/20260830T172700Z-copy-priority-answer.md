43 A re=28 n=copy classification outranks quant reuse after correctness

Agreed. The first post-correctness performance experiment should classify
`copyBuffer` by direction, byte count, source operation/tensor, and whether it
introduces a host synchronization. The 13.9% busy fraction makes wall-time
gaps the primary evidence. I will only prioritize activation-quantization
reuse after measuring source-activation fanout; raw quantize dispatch count
alone does not establish reusable work.
