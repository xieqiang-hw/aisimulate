"""Consolidate scene-1 reports by model without replaying any experiment."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parents[1]
STATE=BASE/'analysis/model_reports'
SOURCES=STATE/'source_documents'
MODEL_FILES={'glm52':'glm52_experiments','minimax':'minimax_experiments'}
BASE_DOCS={'glm52':['experiments_h200_ep8_16x.md'],
           'minimax':['experiments_minimax_h100_64gb_8x.md','experiments_minimax_h100_64gb_32x.md']}
LEGACY=['experiments_h200_ep8_16x.md','experiments_h200_ep8_16x.html','experiments_h200_ep8_16x.workload.html',
        'experiments_h200_ep8_overlay16_speed0p5.md',
        *[f'experiments_minimax_h100_64gb_{part}.{suffix}' for part in ['8x','32x','supplement'] for suffix in ['md','html']],
        'experiments_minimax_h100_64gb_8x.workload.html','experiments_minimax_h100_64gb_32x.workload.html']
COLUMNS=['状态','完成请求','联合 SLA 达标率','P99 TTFT（秒）','平均 TTFT（秒）','TPOT（ms）','HBM 复用','DRAM 直接复用','SSD 恢复复用','总复用率','SSD 累计读','实际耗时','扣排队后平均 TTFT（秒）','扣排队后 P99 TTFT（秒）','内存峰值（GB）']
LABELS={'completed':'已完成','running':'运行中','pending':'待启动','failed':'失败','interrupted':'已中断','skipped':'已跳过'}
KEYS={'sla':2,'p99_ttft':3,'mean_ttft':4,'tpot':5,'hbm':6,'dram':7,'ssd':8,'reuse':9,'ssd_read':10,'adjusted_mean':12,'adjusted_p99':13,'memory':14}
METRICS=[('sla','联合 SLA 达标率','%'),('mean_ttft','平均 TTFT','秒'),('p99_ttft','P99 TTFT','秒'),('adjusted_mean','扣排队后平均 TTFT','秒'),('adjusted_p99','扣排队后 P99 TTFT','秒'),('tpot','TPOT','ms'),('hbm','HBM 复用率','%'),('dram','DRAM 直接复用率','%'),('ssd','SSD 恢复复用率','%'),('reuse','总前缀复用率','%'),('ssd_read','SSD 累计读','TB'),('memory','进程树内存峰值','GB')]


def atomic(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.unified.tmp')
    temp.write_text(text,encoding='utf-8');temp.replace(path)


def ingest_legacy():
    """Route old writers still loaded in memory to the retained source archive."""
    SOURCES.mkdir(parents=True,exist_ok=True)
    moved=[]
    for name in LEGACY:
        p=BASE/name
        if p.is_file() and not p.is_symlink():
            p.replace(SOURCES/name);moved.append(name)
    return moved


def row_key(r):
    return f'{r["factor"]}/{r["speedup"]:g}/{r["budget"]}/{r["dram_gb"]}/{r["read_gbps"]:g}'


def number(text):
    m=re.match(r'[-+]?\d[\d,]*(?:\.\d+)?',str(text))
    return float(m[0].replace(',','')) if m else None


def finish_row(row):
    row['id']=row_key(row)
    row['cells']=(row['cells']+['—']*len(COLUMNS))[:len(COLUMNS)]
    row['status']=row['cells'][0]
    row['values']={k:number(row['cells'][i]) for k,i in KEYS.items()}
    return row


def parse_rows(path,model):
    rows=[];headers=None
    if not path.is_file():return rows
    for line in path.read_text().splitlines():
        if not line.startswith('|'):continue
        cells=[v.strip().replace('**','') for v in line.strip().strip('|').split('|')]
        if '状态' in cells and '完成请求' in cells:
            headers=cells;continue
        if not headers or len(cells)!=len(headers) or not cells[0] or cells[0].startswith('---'):continue
        values=dict(zip(headers,cells))
        factor_text=values.get('标称叠加倍率','')
        if not re.fullmatch(r'\d+×',factor_text):continue
        offset=headers.index('状态')
        read=next((values[k] for k in headers[:offset] if 'SSD' in k and '读带宽' in k),None)
        if read is None:continue
        row=dict(model=model,factor=int(factor_text[:-1]),speedup=float(values.get('speedup','1')),
                 budget=int(values.get('单步预算','16384')),dram_gb=int(values['P 侧 DRAM'].split()[0]),
                 read_gbps=float(read),group='原扫描',source=str(path),cells=cells[offset:])
        rows.append(finish_row(row))
    return rows


def registered_plans():
    p=STATE/'budget_plan_sources.json'
    result=json.loads(p.read_text()) if p.exists() else []
    default=Path('/mnt/e/aisim-results/swe-chat-minimax-h100-64gb-budget-supplement/queued-20261120/sweep.json')
    if default.is_file() and str(default) not in result:result.append(str(default))
    return result


def register_plan(root):
    path=Path(root)/'sweep.json';assert path.is_file()
    values=registered_plans()
    if str(path) not in values:values.append(str(path))
    atomic(STATE/'budget_plan_sources.json',json.dumps(values,indent=2))


def supplement_rows():
    rows=[]
    for filename in registered_plans():
        path=Path(filename)
        if not path.is_file():continue
        plan=json.loads(path.read_text())
        if len(plan.get('cases',[]))<30:continue # exclude the three-case smoke plan
        for c in plan['cases']:
            cells=c.get('metrics_cells') or [LABELS.get(c['status'],c['status'])]+['—']*(len(COLUMNS)-1)
            r=dict(model='minimax',factor=c['factor'],speedup=c['speedup'],budget=c['budget'],dram_gb=c['dram_gb'],
                   read_gbps=c['read_gbps'],group='预算补测',source=str(path),cells=list(cells),
                   detail_available=not c.get('details_pruned',False),error=c.get('error'))
            rows.append(finish_row(r))
    return rows


def signature(model):
    paths=[SOURCES/n for n in BASE_DOCS[model]]+[Path(__file__),Path(__file__).with_name('model_report_dashboard.html')]
    if model=='minimax':paths.extend(Path(p) for p in registered_plans())
    return [[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths if p.is_file()]


def config_info(model):
    shared=dict(deployment='1P1D；P/D 各 8 卡，共 16 卡',parallelism='Attention TP8 / DP1；MoE TP1 / EP8',
                cache='P：HBM + 本地 DRAM + worker_local SSD；D：仅 HBM',ssd='约 56 TB；写 8 GB/s/TP 分片',
                transfer='H2D / D2H 各 25 GB/s/TP；P→D 直接交接 50 GB/s/TP，destination_missing',
                scheduler='P/D 序列上限 32/256；开启 Chunked Prefill',sla='TTFT ≤ 1 秒，且单请求平均 TPOT ≤ 30 ms',
                context='128K（输入与输出合计筛选上限），回放含 output_token_ids',requests=607190,
                source='SWE-chat Claude Code 主会话，verified-interleave 版；会话叠加，保留会话内间隔')
    if model=='glm52':
        shared.update(name='GLM-5.2',model='zai-org/GLM-5.2-FP8',hardware='H200 SXM；使用 H200 性能模型',
                      precision='GLM-5.2-FP8 模型；KV 字节数使用框架解析的 GLM DSA 布局',hbm='P/D 自动 KV 容量，memory_fraction=0.9；块大小 64 token',
                      note='场景 1；不含吞吐承载能力、Prefill 排队诊断及盘框场景。')
    else:
        shared.update(name='MiniMax-M2.5',model='MiniMaxAI/MiniMax-M2.5',hardware='H100 SXM；每卡总 HBM 限 64 GB，计算、带宽及互联仍按 H100',
                      precision='FP8 block 权重 / 计算，BF16 Attention / KV',hbm='P/D 自动 KV 容量，memory_fraction=0.9；预算增大也会改变可用 KV 块数',
                      note='沿用 GLM tokenizer 的长度与前缀关系，作为负载形状对照；不是 MiniMax 原生 tokenizer 重建。只保留 BF16 KV 版本。')
    return shared


def markdown(model,rows,stamp):
    info=config_info(model);counts=Counter(r['status'] for r in rows)
    text=f'# {info["name"]} · SWE-chat 场景 1 实验\n\n统一记录本模型的 DRAM、SSD 带宽及调度预算扫描。{info["note"]}\n\n'
    text+=f'更新时间：{stamp}。共 {len(rows)} 组；'+ '，'.join(f'{k} {v} 组' for k,v in counts.items())+'。\n\n'
    text+='## 1. 配置与数据\n\n| 项目 | 内容 |\n|---|---|\n'
    for key,title in [('model','模型'),('hardware','GPU'),('precision','精度'),('deployment','部署'),('parallelism','并行'),('hbm','HBM KV'),('cache','缓存'),('ssd','SSD'),('transfer','传输'),('scheduler','调度'),('sla','联合 SLA'),('context','上下文')]:text+=f'| {title} | {info[key]} |\n'
    text+=f'\n每组全量 {info["requests"]:,} 条请求。{info["source"]}。speedup 小于 1 表示时间间隔放大，不等于更高的叠加倍率。\n\n'
    text+='扣排队后 TTFT 仅扣除 P 首次准入前调度等待，保留 KV 恢复、后续 chunk、交接与 D 侧等待。原始联合 SLA 不变。GB/TB 采用十进制；复用率按首次准入复用 token 总数／输入 token 总数。\n\n'
    if model=='minimax':text+='预算补测在汇总校验后清理逐请求明细，只保留配置、汇总和日志；表中已完成值来自保存的 metrics_cells，缺失项不从旧实验拼补。\n\n'
    text+='## 2. 实验结果\n\n'
    groups=sorted({(r['budget'],r['factor'],r['speedup']) for r in rows})
    for budget,factor,speed in groups:
        rs=[r for r in rows if (r['budget'],r['factor'],r['speedup'])==(budget,factor,speed)];cs=Counter(r['status'] for r in rs)
        text+=f'### 预算 {budget:,} token · {factor}× 叠加 · speedup={speed:g}\n\n共 {len(rs)} 组，已完成 {cs["已完成"]} 组。\n\n'
        cols=['P 侧 DRAM','SSD 读带宽（GB/s/TP）',*COLUMNS]
        text+='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
        for r in sorted(rs,key=lambda x:(x['read_gbps'],x['dram_gb'])):text+='| '+' | '.join([f'{r["dram_gb"]} GB',f'{r["read_gbps"]:g}',*r['cells']])+' |\n'
        text+='\n'
    if model=='glm52':
        source=SOURCES/'experiments_h200_ep8_16x.md'
        if source.exists():
            body=source.read_text();match=re.search(r'## 2\. 32×.*?(?=\n## |\Z)',body,re.S)
            if match:text+='## 3. 已有分析\n\n'+re.sub(r'^## .*\n','',match[0],count=1).strip()+'\n\n'
    text+='## 4. 更新与来源\n\n'
    text+=f'- 交互页面：[{MODEL_FILES[model]}.html]({MODEL_FILES[model]}.html)。\n'
    text+='- 汇总入口：scripts/refresh_model_reports.py；自动刷新包装程序和日志位于 ~/project/tmp/。\n'
    text+='- 旧分散报告存档于 analysis/model_reports/source_documents/；JSON 记录保存原始来源，不删除实验结果或数据包。\n'
    for source in sorted(set(r['source'] for r in rows)):text+='- 来源：'+chr(96)+source+chr(96)+'。\n'
    return text


def refresh(model='all',budget_root=None,force=False):
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        moved=ingest_legacy()
        if budget_root is not None:register_plan(budget_root)
        changed=[]
        for key in (MODEL_FILES if model=='all' else [model]):
            cache=STATE/(key+'_rows.json');old=json.loads(cache.read_text()) if cache.exists() else {}
            sig=signature(key)
            if not force and old.get('signature')==sig and (BASE/(MODEL_FILES[key]+'.html')).exists():continue
            rows={r['id']:r for r in old.get('rows',[]) if r.get('group')!='预算补测'}
            for name in BASE_DOCS[key]:
                for r in parse_rows(SOURCES/name,key):rows[r['id']]=r
            if key=='minimax':
                for r in supplement_rows():rows[r['id']]=r
            ordered=sorted(rows.values(),key=lambda r:(r['budget'],r['factor'],r['speedup'],r['read_gbps'],r['dram_gb']))
            assert ordered and len({r['id'] for r in ordered})==len(ordered)
            assert all(len(r['cells'])==len(COLUMNS) for r in ordered)
            for r in ordered:
                if r['status']=='已完成':assert number(r['cells'][1])==607190,(r['id'],r['cells'][1])
            stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
            payload=dict(model=key,info=config_info(key),generated=stamp,columns=COLUMNS,metrics=[dict(key=k,label=l,unit=u) for k,l,u in METRICS],rows=ordered)
            atomic(BASE/(MODEL_FILES[key]+'.md'),markdown(key,ordered,stamp))
            template=Path(__file__).with_name('model_report_dashboard.html').read_text();assert template.count('__DATA__')==1
            html=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
            atomic(BASE/(MODEL_FILES[key]+'.html'),html)
            atomic(cache,json.dumps(dict(signature=sig,rows=ordered),ensure_ascii=False,separators=(',',':')))
            changed.append(dict(model=key,rows=len(ordered),statuses=dict(Counter(r['status'] for r in ordered))))
        return dict(updated=changed,archived_legacy=moved)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['all',*MODEL_FILES],default='all');p.add_argument('--force',action='store_true');p.add_argument('--budget-root',type=Path);a=p.parse_args()
    print(json.dumps(refresh(a.model,a.budget_root,a.force),ensure_ascii=False))
