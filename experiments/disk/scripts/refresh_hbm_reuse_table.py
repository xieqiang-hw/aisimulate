from pathlib import Path
import csv,json,re
root=Path.home()/"project"
disk=root/"aisimulate/experiments/disk"
doc=disk/"hbm_reuse_analysis.md"
with (disk/"analysis/p_local_8ssd/assets/metrics.csv").open() as f:
 base={int(r["dram_gb"]):r for r in csv.DictReader(f) if float(r["speedup"])==1}
result=root/"results/glm52-1p1d-local-8ssd-overlay"
keys=(b"input_length",b"first_admission_g1_reused_input_tokens",b"first_admission_g2_direct_reused_input_tokens",b"first_admission_g3_reused_input_tokens")
pat=re.compile(rb'"('+b'|'.join(keys)+rb')":\s*(\d+)')
def metrics(run):
 p=run.parent/"prediction"
 s=(p/"prediction.json").open().read(20000)
 def v(k):return json.JSONDecoder().raw_decode(s.split('"'+k+'":',1)[1].lstrip())[0]
 z=dict.fromkeys(keys,0)
 with (p/"requests.jsonl").open("rb") as f:
  for line in f:
   for k,x in pat.findall(line):z[k]+=int(x)
 return (100*v("goodput_completed_requests")/v("completed_requests"),v("p99_ttft_ms")/1000,*(100*z[k]/z[keys[0]] for k in keys[1:]),100*sum(z[k] for k in keys[1:])/z[keys[0]],v("g3_offload")["read"]["completed_bytes"]/1e12)
def row(gb,label,values):
 return f"| {gb} GB | {label} | "+(" | ".join(f.format(float(v)) for f,v in zip(("{:.3f}%","{:.2f} 秒","{:.3f}%","{:.3f}%","{:.3f}%","{:.3f}%","{:.2f} TB"),values)) if values else "— | — | — | — | — | — | —")+" |"
lines=["## 原速与时间叠加对照","","复用率按首次准入统计，分母为各组全部输入 token。","","| P 侧 DRAM | 负载 | 联合 SLA 达标率 | P99 TTFT | HBM 复用 | DRAM 直接复用 | SSD 恢复复用 | 总复用率 | SSD 累计读 |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
done=0
for gb in (384,256,128):
 lines.append(row(gb,"原速 1×",[base[gb][k] for k in ("sla_percent","p99_ttft_s","g1_pct","g2_pct","g3_pct","total_reuse_pct","ssd_read_tb")]))
 for factor in (2,4,8):
  runs=sorted(result.glob(f"*/p_local_8ssd_p{gb}gb_overlay_{factor}x/*/run.json"))
  runs=[run for run in runs if json.loads(run.read_text())["status"]=="completed"] or runs
  run=runs[-1] if runs else None
  complete=run and json.loads(run.read_text())["status"]=="completed"
  done+=bool(complete)
  lines.append(row(gb,f"{factor}× 叠加"+("" if complete else "（待出结果）"),metrics(run) if complete else None))
lines+=["",f"本轮已完成 {done}/9 组。"]
a,b="<!-- overlay-table:start -->","<!-- overlay-table:end -->"
s=doc.read_text()
section=a+"\n"+"\n".join(lines)+"\n"+b
s=re.sub(re.escape(a)+r".*?"+re.escape(b),lambda _:section,s,flags=re.S) if a in s else s.rstrip()+"\n\n"+section+"\n"
doc.write_text(s)
print(f"Updated {doc}: {done}/9")
