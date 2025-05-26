

优化：

1.将所有的float32运算改成float16的运算

2.将post sample删除，直接取argmax

3.将更改所有的rmsnorm为调用算子

4.统计时间，看看除了forward外的推理时间，到底瓶颈在哪里，发现除了forward外，其他时间只占1%，暂时不用优化

5.更改page attention的分配token_idx的策略，之前是循环分配，这次直接组一个张量分配。时间节省不多，因为只会在第0层调用这个接口

6.更改了kv cache的复制策略，之前会循环，为每一个请求复制kv，现在改成一次性复制所有请求的kv，将batch_size=10的forward耗时从0.0017->0.0010

7.目前整体batch_size=10的时候，推理耗时mha这块计算量比mlp计算量大很多

8.update在decode阶段的时间可以忽略不计
decode time: 0.04577978514134884, update time: 0.0001931358128786087
decode time: 0.04610736854374409, update time: 0.0002020653337240219
decode time: 0.04567588120698929, update time: 0.00020070187747478485
decode time: 0.049862053245306015, update time: 0.00024149753153324127

test/perf.md
