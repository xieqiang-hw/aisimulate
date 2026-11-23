// SPDX-License-Identifier: Apache-2.0
//! Offline request histograms. Every valid completed request is counted once.
use std::{fs::File,io::{BufRead,BufReader},path::Path};
use serde::Deserialize;
use serde_json::{json,Value};
#[derive(Default,Deserialize)]struct Wait{ttft_excluding_prefill_scheduling_ms:Option<f64>,scheduling_wait_ms:Option<f64>,ssd_to_dram_wait_ms:Option<f64>,dram_to_hbm_wait_ms:Option<f64>,kv_wait_ms:Option<f64>}
#[derive(Deserialize)]struct Request{terminal_status:String,ttft_ms:Option<f64>,itl_ms:Option<f64>,e2e_latency_ms:Option<f64>,prefill_wait:Option<Wait>}
fn quantile(v:&[f64],p:f64)->f64{v[((v.len()-1)as f64*p).round()as usize]}
fn summary(mut v:Vec<f64>,total:usize,threshold:Option<f64>)->Value{
 if v.is_empty(){return json!({"count":0,"missing":total});}
 v.sort_unstable_by(f64::total_cmp);let n=v.len();let min=v[0];let max=v[n-1];
 let mut views=serde_json::Map::new();
 for (name,cutoff) in [("full",max),("p99",quantile(&v,0.99)),("p95",quantile(&v,0.95))]{
  let hi=if cutoff>min{cutoff}else{min+1.0};
  let mut counts=vec![0usize;400];let mut included=0;
  for &x in &v{if x>cutoff{break}let i=(((x-min)/(hi-min)*400.0).floor()as usize).min(399);counts[i]+=1;included+=1;}
  assert_eq!(counts.iter().sum::<usize>(),included);
  views.insert(name.into(),json!({"lo":min,"hi":hi,"cutoff":cutoff,"degenerate":cutoff<=min,"counts":counts,"included":included,"excluded":n-included}));
 }
 let sum:f64=v.iter().sum();
 json!({"count":n,"missing":total-n,"mean":sum/n as f64,"min":min,"max":max,"p50":quantile(&v,0.5),"p90":quantile(&v,0.9),"p95":quantile(&v,0.95),"p99":quantile(&v,0.99),"zero_count":v.partition_point(|x|*x==0.0),"threshold":threshold,"threshold_pass":threshold.map(|t|v.partition_point(|x|*x<=t)),"views":views})
}
fn main()->anyhow::Result<()>{
 let a:Vec<_>=std::env::args().collect();anyhow::ensure!(a.len()==3,"usage: request_distribution_summary REQUESTS_JSONL OUTPUT_JSON");
 let mut rows=0usize;let mut completed=0usize;let mut other=0usize;let mut invalid=[0usize;8];
 let mut values:Vec<Vec<f64>>=(0..8).map(|_|Vec::new()).collect();
 let mut reader=BufReader::with_capacity(524288,File::open(&a[1])?);let mut line=String::new();
 loop{line.clear();if reader.read_line(&mut line)?==0{break}let r:Request=serde_json::from_str(&line)?;rows+=1;
  if r.terminal_status!="completed"{other+=1;continue}completed+=1;
  let w=r.prefill_wait.unwrap_or_default();
  if let (Some(t),Some(s),Some(c))=(r.ttft_ms,w.scheduling_wait_ms,w.ttft_excluding_prefill_scheduling_ms){anyhow::ensure!(((t-s).max(0.0)-c).abs()<0.01,"adjusted TTFT identity failed on row {rows}");}
  let fields=[r.ttft_ms.map(|v|v/1000.0),w.ttft_excluding_prefill_scheduling_ms.map(|v|v/1000.0),r.itl_ms,w.scheduling_wait_ms.map(|v|v/1000.0),w.kv_wait_ms.map(|v|v/1000.0),w.ssd_to_dram_wait_ms.map(|v|v/1000.0),w.dram_to_hbm_wait_ms,r.e2e_latency_ms.map(|v|v/1000.0)];
  for (i,x) in fields.into_iter().enumerate(){if let Some(v)=x{if v.is_finite()&&v>=0.0{values[i].push(v)}else{invalid[i]+=1}}}
 }
 let names=["ttft","adjusted_ttft","tpot","scheduling_wait","kv_wait","ssd_wait","h2d_wait","latency"];
 let mut metrics=serde_json::Map::new();
 for (i,v) in values.into_iter().enumerate(){let t=match i{0|1=>Some(1.0),2=>Some(30.0),_=>None};let mut m=summary(v,completed,t);m["invalid"]=json!(invalid[i]);metrics.insert(names[i].into(),m);}
 serde_json::to_writer(File::create(Path::new(&a[2]))?,&json!({"version":1,"rows":rows,"completed":completed,"other_terminal":other,"base_bins":400,"metrics":metrics}))?;Ok(())
}
