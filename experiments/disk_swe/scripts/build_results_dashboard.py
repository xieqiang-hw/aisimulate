from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import argparse

TEMPLATE = r'''
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SWE-chat · KV 缓存实验</title>
<style>
:root{--bg:#f3f4ef;--paper:#fff;--ink:#182d27;--muted:#61736c;--line:#dfe5de;--green:#19745c;--blue:#5279ba;--gold:#bb8540;--radius:18px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 Inter,"Microsoft YaHei","PingFang SC",sans-serif}button,select,input{font:inherit}button,select{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #66a995;outline-offset:3px}
.wrap{max-width:1336px;margin:auto;padding:0 32px}header{padding:22px 0;border-bottom:1px solid var(--line)}.top{display:flex;justify-content:space-between;gap:16px;align-items:center}.brand{font-weight:750;letter-spacing:.03em;font-size:17px}.brand small{font-weight:400;color:var(--muted);margin-left:16px;font-size:13px}.actions{display:flex;gap:8px}.btn{border:1px solid var(--line);background:white;border-radius:9px;padding:8px 13px;color:var(--ink)}.btn:hover{border-color:var(--green)}.hero{padding:34px 0 25px;display:flex;justify-content:space-between;gap:30px;align-items:center}.eyebrow{color:var(--green);font-size:12px;font-weight:700;letter-spacing:.12em}.hero h1{font-size:34px;letter-spacing:-.045em;line-height:1.3;margin:8px 0 12px}.hero p{max-width:660px;margin:0;color:var(--muted)}.sla{background:var(--ink);color:#fff;padding:18px 24px;border-radius:14px;min-width:242px}.sla small{display:block;color:#bbcec6;font-size:12px}.sla strong{font-size:23px;letter-spacing:-.03em;display:block;margin:5px 0}.sla span{font-size:12px;color:#d4e0da}.facts{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;background:#fafbf8;margin-bottom:26px}.fact{padding:16px 22px;border-right:1px solid var(--line)}.fact:last-child{border:0}.fact label{display:block;font-size:12px;color:var(--muted)}.fact strong{font-size:18px}.tabs{display:flex;gap:6px;border-bottom:1px solid var(--line);overflow:auto;margin-bottom:18px}.tab{border:0;background:transparent;white-space:nowrap;padding:13px 20px;border-bottom:3px solid transparent;color:var(--muted)}.tab.active{color:var(--ink);border-bottom-color:var(--green);font-weight:700}.intro{display:flex;gap:20px;align-items:center;justify-content:space-between;margin-bottom:18px}.intro h2{margin:0 0 3px;font-size:22px;letter-spacing:-.03em}.intro p{margin:0;color:var(--muted)}.progress{white-space:nowrap;border:1px solid var(--line);border-radius:9px;padding:7px 12px;color:var(--green);font-size:12px;background:#fff}.filters{display:grid;grid-template-columns:1fr 1fr 1.35fr 1fr;gap:14px;align-items:end;background:white;border:1px solid var(--line);border-radius:14px;padding:17px 20px;margin-bottom:18px}.field label{display:block;font-size:12px;font-weight:650;margin-bottom:5px;color:var(--muted)}select{width:100%;padding:8px 32px 8px 11px;border-radius:7px;border:1px solid var(--line);color:var(--ink);background:#fafbf8}select:disabled{opacity:.6}.check{display:flex;gap:7px;align-items:center;padding-bottom:9px;white-space:nowrap;font-size:12px;color:var(--muted)}.note{margin:10px 0 18px;color:var(--muted);font-size:12px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:17px 20px}.card label{color:var(--muted);font-size:12px;display:block}.card strong{font-size:25px;font-weight:650;letter-spacing:-.04em}.card p{margin:3px 0 0;color:var(--muted);font-size:12px}.chart{background:white;border:1px solid var(--line);border-radius:var(--radius);padding:20px 24px;margin-bottom:18px;min-width:0;position:relative}.charthead{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:10px}.chart h3{font-size:16px;margin:0}.chart .sub{color:var(--muted);font-size:12px;margin:2px 0 0}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend span{display:inline-flex;align-items:center;gap:5px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.canvasholder{position:relative;height:315px}canvas{width:100%;height:100%;display:block}.duo{display:grid;grid-template-columns:1fr 1fr;gap:18px}.duo .canvasholder{height:310px}.insight{padding:17px 22px;background:#e6eee7;border:1px solid #d0dfd3;border-radius:14px;margin:0 0 18px}.insight strong{font-size:13px;display:block;margin-bottom:4px}.insight p{margin:0;font-size:13px}.tip{position:fixed;pointer-events:none;display:none;background:#19332c;color:#fff;border-radius:9px;padding:10px 14px;box-shadow:0 8px 32px #15291c22;font-size:12px;max-width:320px;z-index:10;white-space:pre-line}
details{border:1px solid var(--line);border-radius:14px;background:#fff;margin:0 0 20px}summary{padding:15px 20px;font-weight:650;cursor:pointer}.tablewrap{overflow:auto;max-height:450px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{padding:11px 15px;text-align:right;border-top:1px solid #edf0eb}th{position:sticky;top:0;background:#f7f9f5;color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}.pending{color:#8a968f}.definitions{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:0 20px 18px;color:var(--muted);font-size:12px}.definitions p{margin:0}footer{padding:12px 0 28px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:16px}.notice{font-size:12px;color:var(--muted);padding:0 0 10px}.error{color:#a23526}a{color:var(--green)}@media(max-width:800px){.wrap{padding:0 16px}.hero{align-items:stretch;flex-direction:column;gap:18px}.hero h1{font-size:28px}.facts{grid-template-columns:repeat(2,1fr)}.fact:nth-child(2){border-right:0}.filters{grid-template-columns:1fr 1fr}.duo{grid-template-columns:1fr;gap:0}.cards{gap:8px}.card{padding:13px 12px}.card strong{font-size:21px}.chart{padding:18px 14px}.charthead{flex-direction:column}.definitions{grid-template-columns:1fr}.brand small{display:none}.actions .btn{font-size:12px;padding:7px 9px}footer{flex-direction:column;gap:4px}.canvasholder{height:290px}}@media print{.actions,.filters,.tabs{display:none}.hero{padding:15px 0}.duo{display:block}.chart{break-inside:avoid}.wrap{max-width:none}.chart .canvasholder{height:270px}body{background:white}details .tablewrap{max-height:none}}

.info-panels{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:26px}.info-panel{border:1px solid var(--line);background:#fafbf8;border-radius:var(--radius);padding:20px 24px}.info-panel h2{font-size:18px;margin:0 0 13px;letter-spacing:-.02em}.info-panel dl{margin:0}.info-panel dl>div{display:grid;grid-template-columns:110px 1fr;gap:12px;padding:9px 0;border-top:1px solid #e6ebe4}.info-panel dt{font-size:12px;color:var(--muted);padding-top:1px}.info-panel dd{font-size:13px;margin:0;line-height:1.7}.info-foot{font-size:11px;color:var(--muted);margin:11px 0 0;line-height:1.7}@media(max-width:800px){.info-panels{grid-template-columns:1fr}.info-panel{padding:17px 18px}.info-panel dl>div{grid-template-columns:90px 1fr;gap:10px}}@media print{.info-panel{break-inside:avoid}}
</style>
</head><body>
<header><div class="wrap top"><div class="brand">SWE-chat <small>· KV 缓存实验</small></div><div class="actions"><button class="btn" id="import">导入最新 Markdown</button><button class="btn" id="csv">导出当前 CSV</button></div><input id="file" type="file" accept=".md,.txt" hidden></div></header>
<main class="wrap">
<div class="hero"><div><div class="eyebrow">AISIM / GLM-5.2-FP8</div><h1>DRAM 容量与 SSD 带宽，<br>怎样影响前缀复用？</h1><p>在同一份 Agent 请求上比较缓存容量、会话叠加和 SSD 读带宽。选择实验与组合，查看延迟、SLA 和各层复用的变化。</p></div><div class="sla"><small>联合 SLA · 逐请求判定</small><strong>TTFT ≤ 1 s</strong><span>且平均 TPOT ≤ 30 ms</span></div></div>
<section class="info-panels" aria-label="实验配置与数据集">
<article class="info-panel"><h2>实验配置</h2><dl>
<div><dt>目标模型</dt><dd>GLM-5.2-FP8</dd></div>
<div><dt>硬件与规模</dt><dd>H200 SXM · 1P1D，P/D 各 8 卡，共 16 卡</dd></div>
<div><dt>并行与性能模型</dt><dd>Attention TP8 / DP1 / PP1；MoE TP8 / EP1<br>vLLM 0.24.0 · 算子级耗时估算</dd></div>
<div><dt>P 侧缓存</dt><dd id="configP"></dd></div>
<div><dt>D 侧缓存</dt><dd>HBM 自动容量（memory_fraction=0.9）<br>独立本地 DRAM ≈256 GB</dd></div>
<div><dt>P 侧 SSD</dt><dd id="configSSD"></dd></div>
<div><dt>搬运与调度</dt><dd>H2D 25 GB/s，D2H 32 GB/s<br>P/D 单步预算均为 8,192 token；活跃序列上限分别为 4 / 256</dd></div>
</dl><p class="info-foot">P 侧缓存与 SSD 带宽随下方实验选择更新。容量取近似十进制值，搬运带宽按每 TP 分片计。</p></article>
<article class="info-panel"><h2>数据集</h2><dl>
<div><dt>来源与范围</dt><dd>SWE-chat · Claude Code 主会话<br>原始模型标签统一映射到 GLM-5.2-FP8</dd></div>
<div><dt>回放规模</dt><dd>174,438 条模型请求 · 11,103 个会话<br>491 个可识别用户 ID；3,186 个会话缺少用户 ID</dd></div>
<div><dt>时间范围</dt><dd>2026-01-16 04:30 — 2026-09-04 22:49<br>北京时间，约 232 天；首个响应事件时间近似到达时间</dd></div>
<div><dt>筛选口径</dt><dd>GLM-5.2 tokenizer 重建后，输入＋输出 ≤131,072 token<br>拆出 190,397 条调用，过滤 15,951 条超限请求</dd></div>
<div><dt>格式与块粒度</dt><dd>Mooncake JSONL · trace 块 16 token<br>引擎 KV 块 64 token；回放提供输出长度</dd></div>
<div><dt>当前时间轴</dt><dd id="configTimeline"></dd></div>
</dl><p class="info-foot">按日志可见内容重建；系统提示和完整工具定义未纳入。遇到上下文压缩或调用边界不明确时停止该会话后续提取。叠加保留会话内间隔和越界尾部。</p></article>
</section>
<nav class="tabs" aria-label="实验"><button class="tab" data-mode="baseline">DRAM 基线</button><button class="tab" data-mode="hbm">HBM 容量</button><button class="tab" data-mode="overlay">会话叠加</button><button class="tab active" data-mode="bandwidth">SSD 带宽</button></nav>
<div class="intro"><div><h2 id="title"></h2><p id="description"></p></div><div class="progress" id="progress"></div></div>
<div class="filters"><div class="field"><label for="factor">标称叠加倍率</label><select id="factor"></select></div><div class="field"><label for="dram">P 侧 DRAM</label><select id="dram"></select></div><div class="field"><label for="metric">主图指标</label><select id="metric"><option value="sla">联合 SLA 达标率</option><option value="p99">P99 TTFT</option><option value="mean">平均 TTFT</option><option value="tpot">平均 TPOT</option><option value="ssd">SSD 累计读量</option></select></div><div class="field"><label for="axis">坐标范围</label><select id="axis"><option value="zero">从 0 开始</option><option value="local" selected>最小值附近 · 局部刻度</option></select></div></div>
<div class="notice" id="density"></div>
<div class="cards" id="cards"></div>
<div class="chart"><div class="charthead"><div><h3 id="mainTitle"></h3><p class="sub" id="mainSub"></p></div><div class="legend" id="mainLegend"></div></div><div class="canvasholder"><canvas id="mainChart" role="img" aria-label="所选实验指标对比图"></canvas></div></div>
<div class="insight" aria-live="polite"><strong id="insightTitle">当前对照</strong><p id="insight"></p></div>
<div class="duo"><div class="chart"><div class="charthead"><div><h3>前缀复用来自哪一层</h3><p class="sub">首次准入复用 token ÷ 全部输入 token</p></div></div><div class="legend"><span><i class="dot" style="background:#19745c"></i>HBM</span><span><i class="dot" style="background:#5279ba"></i>DRAM 直接</span><span><i class="dot" style="background:#bb8540"></i>SSD 恢复</span><span><i class="dot" style="background:#e3e8e1"></i>未复用</span></div><div class="canvasholder"><canvas id="reuseChart" role="img" aria-label="HBM DRAM SSD 前缀复用比例"></canvas></div></div><div class="chart"><div class="charthead"><div><h3>累计 SSD 读取</h3><p class="sub">同一批请求的全程读流量，单位 TB；刻度范围跟随上方选项</p></div></div><div class="canvasholder"><canvas id="ssdChart" role="img" aria-label="SSD 累计读取量"></canvas></div></div></div>
<details id="dataDetails"><summary id="tableSummary">查看所选组合的完整数据</summary><div class="tablewrap"><table><thead><tr><th>配置</th><th>状态</th><th>SLA 达标</th><th>P99 TTFT / s</th><th>平均 TTFT / s</th><th>TPOT / ms</th><th>HBM / %</th><th>DRAM / %</th><th>SSD / %</th><th>总复用 / %</th><th>SSD 累计读 / TB</th></tr></thead><tbody id="rows"></tbody></table></div></details>
<details><summary>统计口径与实验边界</summary><div class="definitions"><p><b>延迟与 SLA</b><br>每条请求同时满足 TTFT ≤ 1 秒和平均 TPOT ≤ 30 ms 即达标。TPOT = (E2E − TTFT) ÷ (输出 token 数 − 1)。表中的平均 TPOT 再对请求取平均。仅输出 1 token 的请求跳过 TPOT 判定。</p><p><b>缓存与带宽</b><br>三层复用按输入 token 加权。SSD 恢复包含经 DRAM 中转的复用。累计读量可能包含重复恢复。带宽统一为 GB/s／TP 分片，容量和流量使用十进制 GB、TB。</p><p><b>会话叠加</b><br>每个会话按首条保留请求分段，所有请求统一平移。保留跨切点尾部，因此标称倍率与全程平均密度可能不同，32× 和 64× 的全程平均值尤其受尾部影响。</p><p><b>对照配置</b><br>叠加和带宽实验固定 P 侧 HBM KV 2,048 块。带宽实验固定 P 侧 DRAM 128 GB、SSD 写带宽 8 GB/s／TP 分片；256 GB／8 GB/s 为容量对照。基线和 HBM 容量实验使用各自独立的配置。</p></div></details>
<div class="note" id="sourceStatus"></div><footer><span>数据源 <a href="experiments.md">experiments.md</a> · 内嵌数据快照</span><span id="stamp"></span></footer>
</main><div class="tip" id="tip"></div>
<script id="source" type="application/json">__DATA__</script>
<script>
"use strict";
const $=id=>document.getElementById(id);
const meta=JSON.parse($("source").textContent), colors={128:"#19745c",256:"#5279ba",384:"#bb8540"};
let sourceText=meta.markdown, data=[], mode="baseline", filtered=[];
const metrics={sla:{name:"联合 SLA 达标率",unit:"%",digits:3},p99:{name:"P99 TTFT",unit:"秒",digits:2},mean:{name:"平均 TTFT",unit:"秒",digits:2},tpot:{name:"平均 TPOT",unit:"ms",digits:2},ssd:{name:"SSD 累计读量",unit:"TB",digits:3}};
const descriptions={baseline:["原速 1× · DRAM 容量基线","自动 HBM 容量，SSD 读写各 8 GB/s／TP 分片。"],hbm:["原速 1× · HBM KV 容量扫描","P 侧 DRAM 固定 384 GB，对比自动容量与三档固定 KV 块数。"],overlay:["会话叠加 · 三档 DRAM","P 侧 HBM 固定 2,048 块，SSD 读写各 8 GB/s／TP 分片。"],bandwidth:["SSD 读带宽 · 容量替代对照","P 侧 HBM 固定 2,048 块，SSD 写带宽固定 8 GB/s／TP 分片。"]};
function number(v){if(!v||v.includes("—"))return null; const n=parseFloat(v.replaceAll(",",""));return Number.isFinite(n)?n:null}
function parseMarkdown(text){
 let section=0,header=[],result=[];
 const modes={1:"baseline",2:"hbm",3:"overlay",4:"bandwidth"};
 for(const line of text.split(/\r?\n/)){
  const heading=line.match(/^## (\d+)\./);if(heading){section=+heading[1];header=[];continue}
  if(!line.trim().startsWith("|")||!modes[section])continue;
  const cells=line.trim().replace(/^\||\|$/g,"").split("|").map(x=>x.trim().replaceAll("**",""));
  if(cells.some(x=>x==="状态")&&cells.some(x=>x.includes("联合 SLA"))){header=cells;continue}
  if(!header.length||cells[0].startsWith("---"))continue;
  if(cells.length!==header.length)throw Error("第 "+section+" 节表格列数不一致");
  const row=Object.fromEntries(header.map((h,i)=>[h,cells[i]]));
  const lookup=key=>row[header.find(h=>h.includes(key))];
  const state=lookup("状态");
  if(!state)continue;
  result.push({mode:modes[section],dram:section===2?384:number(lookup("P 侧 DRAM")),
   factor:section<3?1:number(lookup("叠加倍率")),bw:section===4?number(lookup("SSD 读带宽")):8,
   hbm:section===2?(number(lookup("KV 块数"))??"auto"):(section>=3?2048:"auto"),
   status:state,count:number(lookup("完成请求")),sla:number(lookup("联合 SLA")),p99:number(lookup("P99 TTFT")),
   mean:number(lookup("平均 TTFT")),tpot:number(lookup("TPOT")),g1:number(lookup("HBM 复用")),
   g2:number(lookup("DRAM 直接")),g3:number(lookup("SSD 恢复")),total:number(lookup("总复用")),ssd:number(lookup("SSD 累计读"))});
 }
 if(![1,2,3,4].every(n=>result.some(r=>r.mode===modes[n])))throw Error("需要包含第 1～4 节的实验记录");
 if(result.some(r=>r.sla!==null&&(r.sla<0||r.sla>100)))throw Error("SLA 数据超出范围");
 return result;
}
function fmt(v,d=2){return v===null||v===undefined?"—":v.toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d})}
function options(el,values,preferred){el.replaceChildren(...values.map(([value,label])=>new Option(label,value)));if(values.some(x=>String(x[0])===String(preferred)))el.value=preferred}
function configure(){
 const rows=data.filter(r=>r.mode===mode);const factors=[...new Set(rows.map(r=>r.factor))].sort((a,b)=>a-b);
 options($("factor"),[["all","全部倍率"],...factors.map(x=>[String(x),x===1?"原速 1×":x+"× 叠加"])],mode==="bandwidth"?String(factors.at(-1)):"all");
 $("factor").disabled=factors.length===1;if(factors.length===1)$("factor").value=String(factors[0]);
 const drams=[...new Set(rows.map(r=>r.dram))].sort((a,b)=>a-b);
 options($("dram"),[["all","全部 DRAM 档位"],...drams.map(x=>[String(x),x+" GB"])],"all");
 $("dram").disabled=drams.length===1;
 document.querySelectorAll(".tab").forEach(b=>{b.classList.toggle("active",b.dataset.mode===mode);b.setAttribute("aria-pressed",String(b.dataset.mode===mode))});
 render();
}
function label(r){if(mode==="hbm")return r.hbm==="auto"?"自动容量":r.hbm.toLocaleString()+" 块";if(mode==="bandwidth")return r.dram+" GB · "+r.bw+" GB/s";if(mode==="overlay")return r.factor+"× · "+r.dram+" GB";return r.dram+" GB"}
function currentRows(){
 return data.filter(r=>r.mode===mode&&($("factor").value==="all"||r.factor===+$("factor").value)&&($("dram").value==="all"||r.dram===+$("dram").value))
 .sort((a,b)=>mode==="hbm"?((a.hbm==="auto"?1e6:a.hbm)-(b.hbm==="auto"?1e6:b.hbm)):a.factor-b.factor||a.dram-b.dram||a.bw-b.bw);
}
function makeLegend(el,items){el.replaceChildren(...items.map(([text,color])=>{const span=document.createElement("span"),i=document.createElement("i");i.className="dot";i.style.background=color;span.append(i,document.createTextNode(text));return span}))}
function configuration(){
 const dram=$("dram").value==="all"?(mode==="hbm"?"384 GB":mode==="bandwidth"?"128 GB（容量对照为 256 GB）":"128 / 256 / 384 GB"):$("dram").value+" GB";
 const hbm=mode==="baseline"?"自动容量（memory_fraction=0.9）":mode==="hbm"?"自动容量、4,096 / 3,072 / 2,048 块":"2,048 块（整个 TP8 实例约 63.14 GB）";
 $("configP").textContent="HBM "+hbm+"；独立本地 DRAM ≈"+dram;
 $("configSSD").textContent="约 56 TB，worker_local；"+(mode==="bandwidth"?"读带宽 8 / 16 / 32 / 48 / 64 / 128 GB/s":"读带宽 8 GB/s")+"，写带宽 8 GB/s";
 const factor=$("factor").value;
 $("configTimeline").textContent=["overlay","bandwidth"].includes(mode)?(factor==="all"?"比较多个标称叠加倍率；所有叠加 trace 用 speedup=1 回放。":"标称 "+factor+"× 会话叠加；engine speedup=1。"):"原速 1×，按 trace 时间戳回放。";
}
function render(){
 configuration();
 filtered=currentRows();const complete=filtered.filter(r=>r.sla!==null);
 const all=data.filter(r=>r.mode===mode),done=all.filter(r=>r.sla!==null).length;
 $("title").textContent=descriptions[mode][0];$("description").textContent=descriptions[mode][1];
 $("progress").textContent=done+" / "+all.length+" 行已有结果";
 const density={1:1,2:1.987,4:3.676,8:6.801,16:10.488,32:13.695,64:13.695,128:15.336};
 $("density").textContent=["overlay","bandwidth"].includes(mode)?($("factor").value==="all"?"标称倍率按会话分段折叠；全程平均密度会被越界尾部拉低。":"标称 "+$("factor").value+"× · 含尾部的全程平均密度约 "+density[$("factor").value]+"× · 会话内间隔保留"):"每组 174,438 条请求；三档使用相同的原速时间轴。";
 const m=metrics[$("metric").value],best=complete.length?Math.max(...complete.map(r=>r.sla)):null;
 const low=complete.length?Math.min(...complete.map(r=>r.p99)):null;
 const maxRead=complete.length?Math.max(...complete.map(r=>r.ssd)):null;
 $("cards").replaceChildren(...[["最高联合达标率",fmt(best,3)+"%","当前筛选范围"],["最低 P99 TTFT",fmt(low)+" 秒","当前筛选范围"],["最大累计 SSD 读",fmt(maxRead,3)+" TB","当前筛选范围"]].map(x=>{const card=document.createElement("div");card.className="card";const l=document.createElement("label"),v=document.createElement("strong"),p=document.createElement("p");l.textContent=x[0];v.textContent=x[1];p.textContent=x[2];card.append(l,v,p);return card}));
 $("mainTitle").textContent=m.name+"对比";
 $("mainSub").textContent=mode==="bandwidth"?"横轴为 SSD 读带宽；256 GB 对照使用 8 GB/s，图中画为水平虚线。":"横轴按所选实验配置排列，刻度范围可在上方切换。";
 $("mainSub").textContent+=" "+($("axis").value==="local"?"当前为局部刻度，柱形长度由轴下限起算。":"当前坐标从 0 起算。");
 const groups=makeSeries(complete,$("metric").value);
 makeLegend($("mainLegend"),groups.map(x=>[x.name,x.color]));
 plot($("mainChart"),groups,m.unit);
 reuse(complete);ssdBars(complete);insight(complete);
 $("rows").replaceChildren(...filtered.map(r=>{const tr=document.createElement("tr");if(r.sla===null)tr.className="pending";const values=[label(r)+(mode==="bandwidth"&&$("factor").value==="all"?" · "+r.factor+"×":""),r.status,fmt(r.sla,3)+"%",fmt(r.p99),fmt(r.mean),fmt(r.tpot),fmt(r.g1,3),fmt(r.g2,3),fmt(r.g3,3),fmt(r.total,3),fmt(r.ssd,3)];for(const value of values){const td=document.createElement("td");td.textContent=value;tr.append(td)}return tr}));
 $("tableSummary").textContent="查看所选组合的完整数据（"+filtered.length+" 行）";
}
function makeSeries(rows,key){
 const result=[];
 if(mode==="overlay"){
  for(const gb of [128,256,384]){const list=rows.filter(r=>r.dram===gb);if(list.length)result.push({name:gb+" GB DRAM",color:colors[gb],points:list.map(r=>({x:r.factor,label:r.factor+"×",y:r[key],record:r}))})}
 }else if(mode==="bandwidth"){
  for(const factor of [...new Set(rows.map(r=>r.factor))]){
   const a=rows.filter(r=>r.factor===factor&&r.dram===128);
   if(a.length)result.push({name:factor+"× · 128 GB",color:({1:"#936047",2:"#626bc2",4:"#b38d28",8:"#87ad9b",16:"#408d76",32:"#19745c",64:"#174c41",128:"#905ba7"})[factor],points:a.map(r=>({x:r.bw,label:r.bw+"",y:r[key],record:r}))});
   const b=rows.find(r=>r.factor===factor&&r.dram===256);
   if(b)result.push({name:factor+"× · 256 GB / 8 GB/s",color:colors[256],dashed:true,points:[{x:8,label:"8",y:b[key],record:b},{x:Math.max(64,...a.map(r=>r.bw)),label:String(Math.max(64,...a.map(r=>r.bw))),y:b[key],record:b}]});
  }
 }else{
  const sorted=[...rows].sort((a,b)=>mode==="hbm"?(a.hbm==="auto"?1e6:a.hbm)-(b.hbm==="auto"?1e6:b.hbm):a.dram-b.dram);
  for(let i=0;i<sorted.length;i++){const r=sorted[i];result.push({name:label(r),color:colors[r.dram]||colors[128],bar:true,points:[{x:i,label:label(r),y:r[key],record:r}]})}
 }
 return result;
}
const hover=new Map();
function context(canvas){const box=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(box.width*d));canvas.height=Math.max(1,Math.round(box.height*d));const c=canvas.getContext("2d");c.scale(d,d);c.clearRect(0,0,box.width,box.height);c.font='11px "Microsoft YaHei", sans-serif';c.fillStyle="#61736c";hover.set(canvas,[]);return {c,w:box.width,h:box.height}}
function empty(c,w,h){c.textAlign="center";c.fillText("所选组合暂无完整结果",w/2,h/2)}
function axisBounds(values, percent=false){
 const min=Math.min(...values),max=Math.max(...values);
 if($("axis").value!=="local")return [0,percent?100:(max||1)*1.12];
 const padding=Math.max((max-min)*.15,Math.abs(max)*.00001,.0001);
 return [Math.max(0,min-padding),percent?Math.min(100,max+padding):max+padding];
}
function tickFormat(value,range){
 const digits=Math.min(6,Math.max(0,Math.ceil(-Math.log10(range/5))+1));
 return value.toLocaleString("zh-CN",{minimumFractionDigits:digits,maximumFractionDigits:digits});
}
function plot(canvas,series,unit){
 const {c,w,h}=context(canvas),all=series.flatMap(s=>s.points);if(!all.length){empty(c,w,h);return}
 const left=62,right=22,top=25,bottom=50,W=w-left-right,H=h-top-bottom,ys=all.map(p=>p.y);
 let [ymin,ymax]=axisBounds(ys,unit==="%");
 canvas.dataset.axisMin=ymin;canvas.dataset.axisMax=ymax;
 const Y=y=>top+H*(1-((y-ymin)/(ymax-ymin)));
 const xs=[...new Set(all.map(p=>p.x))].sort((a,b)=>a-b);
 const bars=series.every(s=>s.bar),xmin=xs[0],xmax=xs.at(-1);
 const X=x=>bars?left+W*(xs.indexOf(x)+.5)/xs.length:left+(xmax===xmin?W/2:W*(x-xmin)/(xmax-xmin));
 c.fillStyle="#61736c";c.textAlign="left";c.fillText(unit+($("axis").value==="local"?" · 局部刻度":""),left,13);
 for(let i=0;i<=5;i++){const y=ymin+(ymax-ymin)*i/5,py=Y(y);c.strokeStyle="#e8ede6";c.lineWidth=1;c.setLineDash([]);c.beginPath();c.moveTo(left,py);c.lineTo(w-right,py);c.stroke();c.textAlign="right";c.fillText(tickFormat(y,ymax-ymin),left-9,py+4)}
 const labels=new Map(all.map(p=>[p.x,p.label]));c.textAlign="center";
 xs.forEach(x=>c.fillText(labels.get(x),X(x),h-24));
 for(const s of series){
  c.strokeStyle=s.color;c.fillStyle=s.color;c.lineWidth=2.5;c.setLineDash(s.dashed?[6,5]:[]);
  if(s.bar){const p=s.points[0],bw=Math.min(66,W/xs.length*.46);c.fillRect(X(p.x)-bw/2,Y(p.y),bw,top+H-Y(p.y))}
  else{c.beginPath();s.points.forEach((p,i)=>{if(i)c.lineTo(X(p.x),Y(p.y));else c.moveTo(X(p.x),Y(p.y))});c.stroke()}
  s.points.forEach(p=>{if(!s.dashed){c.beginPath();c.arc(X(p.x),Y(p.y),4,0,Math.PI*2);c.fill()}hover.get(canvas).push({x:X(p.x),y:Y(p.y),r:p.record,text:s.name+"\n"+(s.dashed?"容量对照":p.label)+" · "+fmt(p.y,metrics[$("metric").value].digits)+" "+unit})});
 }c.setLineDash([]);
}
function rowLabel(r){if(mode==="bandwidth")return ($("factor").value==="all"?r.factor+"× ":"")+r.dram+"G / "+r.bw;if(mode==="overlay")return r.factor+"× / "+r.dram+"G";return label(r)}
function reuse(rows){
 const canvas=$("reuseChart"),{c,w,h}=context(canvas);if(!rows.length){empty(c,w,h);return}
 const left=130,right=20,top=22,bottom=22,W=w-left-right,H=h-top-bottom,step=H/rows.length;
 c.textAlign="center";for(let n=0;n<=100;n+=25){const x=left+W*n/100;c.strokeStyle="#edf0eb";c.beginPath();c.moveTo(x,top);c.lineTo(x,h-bottom);c.stroke();c.fillText(n+"%",x,13)}
 rows.forEach((r,i)=>{const y=top+step*i+step*.2,bh=step*.6;c.textAlign="right";c.fillStyle="#61736c";c.font=(rows.length>12?'9':'11')+'px "Microsoft YaHei", sans-serif';c.fillText(rowLabel(r),left-9,y+bh/2+4);let x=left;const vals=[r.g1,r.g2,r.g3,Math.max(0,100-r.g1-r.g2-r.g3)];vals.forEach((v,j)=>{c.fillStyle=["#19745c","#5279ba","#bb8540","#e3e8e1"][j];c.fillRect(x,y,W*v/100,bh);x+=W*v/100});hover.get(canvas).push({x:left+W/2,y:y+bh/2,text:rowLabel(r)+"\nHBM "+fmt(r.g1,3)+"% · DRAM "+fmt(r.g2,3)+"%\nSSD "+fmt(r.g3,3)+"% · 总复用 "+fmt(r.total,3)+"%",wide:true})});
}
function ssdBars(rows){
 const canvas=$("ssdChart"),{c,w,h}=context(canvas);if(!rows.length){empty(c,w,h);return}
 const left=130,right=65,top=34,bottom=22,W=w-left-right,H=h-top-bottom,step=H/rows.length;
 const [min,max]=axisBounds(rows.map(r=>r.ssd)),range=max-min;
 const X=value=>left+W*(value-min)/range;
 canvas.dataset.axisMin=min;canvas.dataset.axisMax=max;
 c.textAlign="left";c.fillText($("axis").value==="local"?"TB · 局部刻度，柱形由轴下限起算":"TB · 从 0 起算",left,12);
 for(let i=0;i<=4;i++){
  const value=min+range*i/4,x=X(value);
  c.strokeStyle="#edf0eb";c.beginPath();c.moveTo(x,top);c.lineTo(x,h-bottom);c.stroke();
  c.fillStyle="#61736c";c.textAlign="center";c.fillText(tickFormat(value,range),x,27);
 }
 rows.forEach((r,i)=>{
  const y=top+i*step+step*.2,bh=step*.6;
  c.fillStyle="#61736c";c.textAlign="right";c.font=(rows.length>12?'9':'11')+'px "Microsoft YaHei", sans-serif';
  c.fillText(rowLabel(r),left-9,y+bh/2+4);
  c.fillStyle=colors[r.dram];c.fillRect(left,y,X(r.ssd)-left,bh);
  c.fillStyle="#61736c";c.textAlign="left";c.fillText(fmt(r.ssd,3),X(r.ssd)+6,y+bh/2+4);
  hover.get(canvas).push({x:left+W/2,y:y+bh/2,text:rowLabel(r)+"\n累计 SSD 读取 "+fmt(r.ssd,3)+" TB",wide:true});
 });
}
function insight(rows){
 if(!rows.length){$("insight").textContent="所选范围还没有完整结果。可在下方明细表查看待完成组合。";return}
 if(mode==="bandwidth"&&$("factor").value!=="all"){
  const b=rows.find(r=>r.bw===8&&r.dram===128),ref=rows.find(r=>r.dram===256),high=rows.filter(r=>r.dram===128).sort((a,b)=>b.bw-a.bw)[0];
  if(b&&high&&high.bw>b.bw){$("insight").textContent="128 GB DRAM 的读带宽从 8 提到 "+high.bw+" GB/s，联合达标率变化 "+fmt(high.sla-b.sla,3)+" 个百分点，P99 TTFT 从 "+fmt(b.p99)+" 秒变为 "+fmt(high.p99)+" 秒。"+(ref?"同倍率 256 GB / 8 GB/s 对照的达标率为 "+fmt(ref.sla,3)+"%，P99 TTFT 为 "+fmt(ref.p99)+" 秒。":"");return}
 }
 const min=Math.min(...rows.map(r=>r.sla)),max=Math.max(...rows.map(r=>r.sla));
 $("insight").textContent="当前 "+rows.length+" 个已有结果组合的联合 SLA 达标率为 "+fmt(min,3)+"%～"+fmt(max,3)+"%。结合下方分层复用与 SSD 读量，可查看性能变化是否伴随缓存复用来源转移。";
}
for(const canvas of document.querySelectorAll("canvas")){
 canvas.addEventListener("mousemove",e=>{const box=canvas.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top,points=hover.get(canvas)||[];let best=null,dist=Infinity;for(const p of points){const d=p.wide?Math.abs(y-p.y):Math.hypot(x-p.x,y-p.y);if(d<dist){dist=d;best=p}}const tip=$("tip");if(!best||dist>(best.wide?14:28)){tip.style.display="none";return}tip.textContent=best.text;tip.style.display="block";tip.style.left=Math.max(8,Math.min(e.clientX+12,innerWidth-tip.offsetWidth-12))+"px";tip.style.top=Math.max(8,Math.min(e.clientY+12,innerHeight-tip.offsetHeight-12))+"px"});
 canvas.addEventListener("mouseleave",()=>{$("tip").style.display="none"});
}
document.querySelectorAll(".tab").forEach(b=>b.addEventListener("click",()=>{mode=b.dataset.mode;configure()}));
["factor","dram","metric","axis"].forEach(id=>$(id).addEventListener("change",render));
$("import").addEventListener("click",()=>$("file").click());
$("file").addEventListener("change",async e=>{const file=e.target.files[0];if(!file)return;try{const text=await file.text();const next=parseMarkdown(text);sourceText=text;data=next;$("sourceStatus").className="note";$("sourceStatus").textContent="已导入 "+file.name;$("stamp").textContent="本次导入 "+new Date().toLocaleString("zh-CN");configure()}catch(err){$("sourceStatus").className="note error";$("sourceStatus").textContent="导入失败："+err.message}finally{e.target.value=""}});
$("csv").addEventListener("click",()=>{const keys=["mode","factor","dram","hbm","bw","status","count","sla","p99","mean","tpot","g1","g2","g3","total","ssd"];const content="\ufeff"+[keys,...filtered.map(r=>keys.map(k=>r[k]??""))].map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(",")).join("\r\n");const url=URL.createObjectURL(new Blob([content],{type:"text/csv;charset=utf-8"}));const a=document.createElement("a");a.href=url;a.download="swe-chat-"+mode+".csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
let timer;window.addEventListener("resize",()=>{clearTimeout(timer);timer=setTimeout(render,120)});
try{data=parseMarkdown(sourceText);$("sourceStatus").textContent="已载入本轮 "+data.length+" 行记录。后台更新后刷新浏览器可读取最新汇总。";$("stamp").textContent="页面生成于 "+meta.generated;configure()}catch(err){$("sourceStatus").className="note error";$("sourceStatus").textContent=err.message}
</script>
</body></html>
'''
def build(sweep=None, markdown_path=None, output=None):
    root = Path(__file__).resolve().parents[1]
    markdown_path = Path(markdown_path) if markdown_path else root / "experiments.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.dumps({
        "markdown": markdown,
        "generated": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output = Path(output) if output else root / "experiments.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    temporary.replace(output)
    print(output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Build the results dashboard")
    parser.add_argument("--sweep", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.sweep, args.markdown, args.output)


if __name__ == "__main__":
    main()
