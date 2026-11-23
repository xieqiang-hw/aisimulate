"""Build an offline report for two Prefill scheduling examples from saved traces."""
from pathlib import Path
import json
import html

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE / 'analysis/prefill_queue_diagnosis'
DATA = json.loads((FOLDER / 'window_data.json').read_text())
EXTENDED = json.loads((FOLDER / 'extended_window_data.json').read_text())
DATA.update(iterations=EXTENDED['iterations'],states=EXTENDED['states'],requests=EXTENDED['requests'],
            extended_start_ms=EXTENDED['start_ms'],extended_end_ms=EXTENDED['end_ms'],extended_center_ms=EXTENDED['center_ms'])
from prepare_prefill_main_workload import prepare
MAIN = prepare()
DATA['main_workload'] = MAIN
A = (683087834.9549372, 683100433.8851472)
B = (683014908.181065, 683017350.819919)
PASSES = [r for r in DATA['iterations'] if r[5] > 0]


def queue_at(t):
    return sum(q['arrival_time_ms'] <= t < q['prefill_admit_ms'] for q in DATA['requests'])


def state_at(t):
    state = {}
    for at, _, value in DATA['states']:
        if at > t + .0001:
            break
        state = value
    return state


def static_figure(name, bounds, title):
    lo, hi = bounds
    left, right = 85, 1130
    x = lambda t: left + (t - lo) / (hi - lo) * (right - left)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="640" viewBox="0 0 1180 640">',
             '<rect width="1180" height="640" rx="18" fill="#fff"/>',
             '<style>text{font-family:Microsoft YaHei,Segoe UI,sans-serif;fill:#23443b;font-size:14px}</style>',
             f'<text x="32" y="36" style="font-size:21px;font-weight:bold">{html.escape(title)}</text>']
    def text(a, b, s, extra=''):
        parts.append(f'<text x="{a}" y="{b}" {extra}>{html.escape(str(s))}</text>')
    def line(a,b,c,d,color='#e2e9e1',extra=''):
        parts.append(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="{color}" {extra}/>')
    text(left, 72, '每轮新计算 token 数（条宽 = 执行时长；不是 token/s）')
    top, bottom = 100, 265
    y = lambda n: bottom - n / 18022.4 * (bottom-top)
    for v in [0,4096,8192,12288,16384]:
        line(left,y(v),right,y(v))
        text(left-12,y(v)+5,f'{v:,}','text-anchor="end"')
    for r in PASSES:
        if r[4] <= lo or r[3] >= hi:
            continue
        a,b=max(lo,r[3]),min(hi,r[4]); color='#bc712d' if r[6]==974 else '#537bb2' if r[5]==3 and bounds==B else '#157d69'
        parts.append(f'<rect x="{x(a)}" y="{y(r[6])}" width="{x(b)-x(a)}" height="{bottom-y(r[6])}" fill="{color}" opacity=".82" stroke="white"/>')
        if x(b)-x(a)>55:
            text((x(a)+x(b))/2,max(92,y(r[6])-8),f'{r[6]:,} · BS {r[5]}','text-anchor="middle"')
    line(left,y(16384),right,y(16384),'#8e9b91','stroke-dasharray="5 4"')
    events={}
    for q in DATA['requests']:
        for t,delta in [(q['arrival_time_ms'],1),(q['prefill_admit_ms'],-1)]:
            if lo<t<=hi:events[t]=events.get(t,0)+delta
    count=queue_at(lo);maximum=count
    for t,delta in sorted(events.items()):
        count+=delta;maximum=max(maximum,count)
    qmax=max(4, int((maximum*1.2+3.999)//4)*4); qt,qb=370,535
    qy=lambda n:qb-n/qmax*(qb-qt)
    text(left,340,'P 首次准入前等待请求数（逐请求事件，包含调度与 KV 恢复等待）')
    for i in range(5):
        n=qmax*i/4;line(left,qy(n),right,qy(n));text(left-12,qy(n)+5,f'{n:g}','text-anchor="end"')
    count=queue_at(lo);path=f'M {left} {qy(count)}'
    for t,delta in sorted(events.items()):
        path+=f' H {x(t)} V {qy(count+delta)}';count+=delta
    path+=f' H {right}'
    parts.append(f'<path d="{path}" fill="none" stroke="#157d69" stroke-width="2.5"/>')
    for i in range(6):
        t=lo+(hi-lo)*i/5
        for b in [bottom,qb]:text(x(t),b+25,f'{(t-lo)/1000:.2f}','text-anchor="middle"')
    if bounds==B:
        newcomers=[q for q in DATA['requests'] if B[0]<q['arrival_time_ms']<683016777.460214 and abs(q['prefill_admit_ms']-683016974.424702)<.01]
        for q in newcomers:
            for a,b in [(top,bottom),(qt,qb)]:line(x(q['arrival_time_ms']),a,x(q['arrival_time_ms']),b,'#bc712d','stroke-dasharray="3 5" opacity=".6"')
        text(left,606,'橙色虚线：三条新请求到达；尾段结束时三条请求同时首次准入。')
    else:text(left,606,'普通队列快照中的等待数为 2、3、3、8、10；与下图总等待口径不同。')
    text(right,580,'横轴：距当前视图起点的秒数；上下图使用同一时间轴。','text-anchor="end"')
    parts.append('</svg>')
    (FOLDER/name).write_text('\n'.join(parts),encoding='utf-8')


def static_main_figure():
    source=MAIN['rows'];groups=[]
    for i in range(0,len(source),12):
        rows=source[i:i+12];count=sum(r[0] for r in rows)
        groups.append(dict(start=i*300000,end=min((i+12)*300000,MAIN['end_ms']),maximum=max((r[2] or 0) for r in rows) if count else None,mean=sum(r[1] for r in rows)/count if count else None,peak=max(r[7] for r in rows),last=rows[-1][8]))
    left,right=85,1130;x=lambda t:left+t/MAIN['end_ms']*(right-left)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="640" viewBox="0 0 1180 640">','<rect width="1180" height="640" rx="18" fill="white"/>','<style>text{font-family:Microsoft YaHei,Segoe UI,sans-serif;fill:#23443b;font-size:14px}</style>','<text x="32" y="36" style="font-size:21px;font-weight:bold">原始 8×、P DRAM 256 GB：主要负载阶段（每小时聚合）</text>']
    def text(a,b,s,extra=''):
        parts.append(f'<text x="{a}" y="{b}" {extra}>{html.escape(str(s))}</text>')
    for top,bottom,max_value,ticks,title,series in [(100,265,18022.4,[0,4096,8192,12288,16384],'每轮新算 token：橙色 = 桶内最大值，绿色 = 计算批次平均值',[('maximum','#bc712d'),('mean','#157d69')]),(370,535,16,[0,4,8,12,16],'P 首次准入前等待：橙色 = 桶内峰值，绿色 = 桶末值',[('peak','#bc712d'),('last','#157d69')])]:
        y=lambda v:bottom-v/max_value*(bottom-top)
        text(left,top-25,title)
        for v in ticks:
            parts.append(f'<line x1="{left}" x2="{right}" y1="{y(v)}" y2="{y(v)}" stroke="#e2e9e1"/>')
            text(left-12,y(v)+5,f'{v:,}','text-anchor="end"')
        for key,color in series:
            path='';connected=False
            for g in groups:
                if g[key] is None:connected=False;continue
                path+=(' L' if connected else ' M')+f' {x(g["start"])} {y(g[key])} L {x(g["end"])} {y(g[key])}';connected=True
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        for i in range(6):
            t=MAIN['end_ms']*i/5;text(x(t),bottom+25,f'{t/86400000:.2f}','text-anchor="middle"')
    text(right,580,'横轴：距离 trace 起点的天数；上下图时间轴一致。','text-anchor="end"')
    text(left,609,'保留全部五分钟底层分桶；均值排除空调度轮次；等待峰值来自逐请求事件，不随机抽点。')
    parts.append('</svg>')
    (FOLDER/'main_workload.svg').write_text('\n'.join(parts),encoding='utf-8')


def build():
    budget=[r for r in PASSES if A[0]-.001<=r[2]<A[1]-.001]
    tail=[r for r in PASSES if B[0]-.001<=r[2]<B[1]-.001]
    assert len(budget)==5 and all(r[6]==16384 and r[5]==1 for r in budget)
    assert [r[6] for r in tail]==[16384,974,1336] and [r[5] for r in tail]==[1,1,3]
    newcomers=sorted([q for q in DATA['requests'] if B[0]<q['arrival_time_ms']<tail[0][4] and abs(q['prefill_admit_ms']-tail[1][4])<.01],key=lambda q:q['arrival_time_ms'])
    assert len(newcomers)==3 and all(q['kv_wait_ms']==0 for q in newcomers)
    for t in [A[0],A[1],B[0],tail[1][2],tail[1][4]]:
        assert queue_at(t)>=0
    static_main_figure()
    static_figure('token_budget.svg',A,'情况一：连续五轮 BS = 1，但每轮 token 预算已经用完')
    static_figure('async_tail.svg',B,'情况二：974-token 尾段未满，但新请求只能等后续批次')
    template=Path(__file__).with_name('prefill_queue_dashboard.html').read_text()
    assert template.count('__DATA__')==1
    page=template.replace('__DATA__',json.dumps(DATA,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    (BASE/'prefill_queue_diagnosis.html').write_text(page,encoding='utf-8')
    rows='\n'.join(f'| {i+1} | 1／32 | 16,384／16,384 | {state_at(r[2])["waiting_admission_requests"]} | {(r[4]-r[3])/1000:.3f} |' for i,r in enumerate(budget))
    arrivals='\n'.join(f'| {i+1} | {int(q["input_length"]):,} | {(q["arrival_time_ms"]-B[0])/1000:.3f} | {(q["prefill_admit_ms"]-B[0])/1000:.3f} | {q["scheduling_wait_ms"]/1000:.3f} |' for i,q in enumerate(newcomers))
    md=f'''# 原始 8× 下的 Prefill 排队：两个具体案例

本记录解释两种现象：BS 未满但单轮 token 预算已满；以及单轮 token 预算也未满，但异步调度已提交的批次不再接收新到请求。两种现象均来自已有实验的原始记录，没有重跑或修改引擎。

交互图表见 [独立 HTML](prefill_queue_diagnosis.html)。HTML 内嵌数据，可单文件离线打开；本文的 SVG 图片位于 analysis/prefill_queue_diagnosis，转发 Markdown 时需一并保留。

## 1. 实验与图表口径

- 原始 8× 会话叠加包，时间压缩系数 **1**，不是再压缩 8 倍的实验。
- GLM-5.2-FP8、H200 SXM、1P1D、各 TP8／EP8；P DRAM 256 GB，SSD 读 2 GB/s／TP 分片、写 8 GB/s／TP 分片；D 仅 HBM。
- P 单轮预算 16,384 token，序列上限 32，HBM KV 容量 6,103 块，块大小 64 token。
- 原五分钟窗口：第 7.9027778–7.9062500 天，对应仿真时间 682,800.000–683,100.000 秒。为画完情况一的第五批，数据范围延长至 683,100.434 秒；不将延长段计入原五分钟汇总。
- 图一：每批次新计算 token 数。横向宽度为真实计算时长，纵向高度为该批次新算 token 数；不是 token/s，面积也不表示累计 token。横虚线为整批 16,384-token 预算。
- 图二：P 首次准入前等待请求数，按逐请求到达与首次准入事件计算，包含调度与 KV 恢复等待；不等于 PD 全部未完成请求数。
- 表格中的“普通队列等待”取调度后状态快照，排除了连接器等待等状态，与图二口径不同。同一时刻的到达和准入事件合并处理，不制造零时长积压。

## 2. 主要负载阶段全程图

![主要负载阶段的新计算与等待](analysis/prefill_queue_diagnosis/main_workload.svg)

主要负载阶段为前 **{MAIN['main_days']:.3f} 天**。默认每小时展示一个聚合桶；HTML 可切换每 15 分钟或每 5 分钟，保留全部 {len(MAIN['rows']):,} 个底层五分钟桶，不使用随机抽点。

| 统计项 | 主要负载阶段 |
|---|---:|
| 实际计算批次数 | {MAIN['compute_passes']:,} |
| 实际新算 token 总量（按批次计算开始归属） | {MAIN['main_new_tokens']:,} |
| 用满 16,384-token 预算的批次占比 | {MAIN['full_budget_passes']/MAIN['compute_passes']*100:.2f}% |
| P 首次准入前等待峰值 | {MAIN['main_waiting_peak']} 条 |

全程上图展示桶内**每轮新算 token 最大值与平均值**；均值由总新算 token 数除以真实计算批次数得到，合并时间桶时按批次数加权，排除无计算的调度轮次。桶内没有计算时留空，不画为 0-token 计算。这里的横向宽度表示聚合桶长度，不再表示单批执行时长；单批耗时请看后面的秒级案例。

全程下图展示**桶内等待峰值和桶末等待数**。峰值根据所有逐请求到达、首次准入事件计算，不会因为低频抽样漏掉短时积压。桶末取结束前一瞬，最后一个不足整桶的时间段按实际长度处理。

这两个全程图适合发现哪些时段频繁出现满预算批次、哪些时段有突发等待。它们不能证明两个桶内峰值发生在同一瞬间，更不能据此认定每次等待都由 token 预算造成。全程图与秒级案例互补使用。

HTML 的“选择观察窗口”还可切换到“五分钟中心 ±1 小时”：以原五分钟的中点为中心，左右各一小时，总共两小时。横轴改用相对中心的分钟数，绿色背景标记原五分钟。该视图使用实际逐轮计算与逐请求等待事件，不做随机抽样；提取数据见 [extended_window_data.json](analysis/prefill_queue_diagnosis/extended_window_data.json)。

## 3. 情况一：BS 为 1，但整批 token 预算已满

![连续五轮的实际新算 token 与等待请求数](analysis/prefill_queue_diagnosis/token_budget.svg)

观察区间：仿真时间 683,087.835–683,100.434 秒，持续约 12.599 秒。上下图共用时间轴，便于直接对齐等待出现和每轮执行的关系。

| 轮次 | 实际 BS／上限 | 新算 token／预算 | 调度后普通队列等待 | 本轮耗时（秒） |
|---|---:|---:|---:|---:|
{rows}

16,384 是**整个批次**的 token 预算，不是每个请求各有一份。当前调度器先安排已经运行的请求，一个长请求可以独自耗尽整轮预算。Chunked Prefill 允许长输入分轮计算，但不保证每轮把预算分给多个请求；因此 BS 未达 32，也不能继续向本轮加入需要计算的新请求。

一条输入为 102,948 token 的请求在这段连续计算中到达，首次准入前 KV 等待为 0，调度等待约 **10.511 秒**。这是本案例中 token 预算和调度顺序影响响应的直接例子，不能由此认定 GPU 算力利用率已达到 100%。

## 4. 情况二：974-token 尾段未满，但批次已提前提交

![异步批次的实际执行及三个新请求等待](analysis/prefill_queue_diagnosis/async_tail.svg)

原请求输入 17,486 token，复用 128 token，新计算量正好是：

`17,486 − 128 = 16,384 + 974`。

| 批次 | 新算 token | BS | 本轮耗时（秒） |
|---|---:|---:|---:|
| 原请求主要 chunk | 16,384 | 1 | 1.869 |
| 原请求剩余尾段 | 974 | 1 | 0.197 |
| 三条新请求 | 1,336 | 3 | 0.376 |

以下时刻均以主要 chunk 开始为 0；三条请求在第一批执行期间陆续到达，尾段结束后在同一时刻首次准入，KV 恢复等待均为 0。

| 新请求 | 输入 token | 到达（秒） | 首次准入（秒） | 调度等待（秒） |
|---|---:|---:|---:|---:|
{arrivals}

尾段执行时，BS 为 1／32，新算量为 974／16,384，活跃 HBM 块为 274／6,103，SSD 待读、H2D 加载任务均为 0。这轮无法继续合批的解释是当前 AISim 的异步调度准入规则，而不是本轮 token 或 HBM 容量耗尽。

当前 vLLM 后端默认异步批次深度为 2：忙碌时将下一批视为已提交。到达时已经提交的批次不能再被该请求修改；等待队列使用 admissible_from_pass 检查允许准入的轮次。该例三条请求被推迟到尾段之后，额外延后约 **197 ms**。这些规则来自 AISim 当前实现，尚未通过实机验证其与对应 vLLM 版本是否完全一致。

## 5. 如何理解低平均忙碌占比与排队

原五分钟窗口 P 计算约 49.37 秒，忙碌占比约 16.46%；普通准入队列非空约 12.36 秒，按状态快照还原，这些普通队列等待时段与 P 计算重叠。五分钟平均值混合了长时间空闲与短时间连续计算，不能据此判断某条请求到达时是否能够立即执行。

上述两例说明：**序列数上限、单轮 token 预算和异步批次准入时机是不同约束。** 它们可以使有潜在硬件余量的系统仍产生等待。它们不能量化 GPU 算力利用率，也不能解释全部长尾 TTFT；是否应调整预算或异步调度，需要单因素对照实验。

## 6. 复核与生成

- 源结果目录：`{DATA['source']}`。
- 计算批次：iteration_metrics.jsonl.gz，使用 compute_start_ms、end_ms、prefill_bs、prefill_new_tokens；无计算批次不画成工作条。
- 调度状态：scheduler_states.jsonl.gz，边界为调度后、发出执行前。
- 等待曲线：requests.jsonl，`Q(t) = count(arrival_time_ms ≤ t < prefill_admit_ms)`。
- 提取数据：[window_data.json](analysis/prefill_queue_diagnosis/window_data.json)。保留的是选定窗口数据，不是全量请求副本。
- 生成入口：[build_prefill_queue_diagnosis.py](scripts/build_prefill_queue_diagnosis.py)；页面模板：[prefill_queue_dashboard.html](scripts/prefill_queue_dashboard.html)。在仓库根目录执行 `~/.venvs/aisim/bin/python experiments/disk_swe/scripts/build_prefill_queue_diagnosis.py` 可重新生成 MD、HTML、SVG。
- 调度源码：[准入轮次与等待队列](../../crates/core/src/engine/scheduler/vllm/core.rs)；[异步批次深度默认解析](../../crates/core/src/engine/config.rs)。
'''
    (BASE/'prefill_queue_diagnosis.md').write_text(md,encoding='utf-8')
    (FOLDER/'validation.json').write_text(json.dumps(dict(budget_passes=len(budget),async_passes=len(tail),new_requests=len(newcomers),time_compression=1,checks=['five full 16384-token BS1 batches','tail sizes 16384+974, then BS3/1336','three new requests share admission boundary','three requests have zero KV recovery wait','queue counts non-negative at key boundaries']),indent=2))
    from prepare_prefill_comparison import build as build_comparison
    build_comparison()
    print('Generated prefill_queue_diagnosis.md / .html and two SVG figures')


if __name__=='__main__':
    build()
