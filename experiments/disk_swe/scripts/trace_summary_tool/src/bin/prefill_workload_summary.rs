// SPDX-License-Identifier: Apache-2.0
//! Bounded-memory, offline Prefill batch aggregation; does not change replay.
use std::{fs::File,io::{BufRead,BufReader},path::Path};
use flate2::read::GzDecoder;
use serde_json::json;
type Row=(usize,u64,f64,f64,f64,usize,usize,usize,usize,usize,usize,usize,usize);
#[derive(Default,Clone)]
struct Bin{count:u64,tokens:u64,max:usize,full:u64,max_bs:usize,busy_ms:f64}
fn main()->anyhow::Result<()>{
 let a:Vec<_>=std::env::args().collect();
 anyhow::ensure!(a.len()==4,"usage: prefill_workload_summary INPUT_DIR OUTPUT_JSON END_MS");
 let end:f64=a[3].parse()?;
 let dir=Path::new(&a[1]);
 let meta:serde_json::Value=serde_json::from_reader(File::open(dir.join("scheduler_trace_metadata.json"))?)?;
 let sid=meta["streams"].as_array().unwrap().iter().position(|x|x["role"]=="prefill").unwrap();
 let budget=meta["streams"][sid]["max_batched_tokens"].as_u64().unwrap()as usize;
 let mut bins=vec![Bin::default();(end/300000.0).ceil()as usize];
 let mut r=BufReader::with_capacity(262144,GzDecoder::new(File::open(dir.join("iteration_metrics.jsonl.gz"))?));
 let mut line=String::new();let mut total=0u64;let mut main_tokens=0u64;
 loop{
  line.clear();if r.read_line(&mut line)?==0{break}
  // All records use the compact v2 tuple schema. Parse and price nothing here.
  let v:Row=serde_json::from_str(&line)?;
  if v.0!=sid||v.5==0||v.6==0{continue}
  total+=v.6 as u64;
  if v.3<end{
   let b=&mut bins[(v.3/300000.0)as usize];b.count+=1;b.tokens+=v.6 as u64;
   b.max=b.max.max(v.6);b.full+=u64::from(v.6==budget);b.max_bs=b.max_bs.max(v.5);
   main_tokens+=v.6 as u64;
  }
  let mut t=v.3;
  while t<v.4.min(end){let i=(t/300000.0)as usize;let stop=v.4.min(end).min((i+1)as f64*300000.0);bins[i].busy_ms+=stop-t;t=stop;}
 }
 let rows:Vec<_>=bins.iter().map(|b|json!([b.count,b.tokens,if b.count>0{Some(b.max)}else{None},if b.count>0{Some(b.tokens as f64/b.count as f64)}else{None},b.full,b.max_bs,b.busy_ms])).collect();
 let out=json!({"schema_version":1,"bin_ms":300000,"end_ms":end,"token_budget":budget,"all_recorded_prefill_tokens":total,"main_tokens_by_batch_start":main_tokens,"columns":["compute_passes","new_tokens","max_tokens_per_pass","mean_tokens_per_pass","full_budget_passes","max_bs","compute_ms_clipped"],"rows":rows});
 serde_json::to_writer(File::create(&a[2])?,&out)?;
 println!("{} bins; {} main Prefill tokens; {} recorded tokens in all phases of trace",rows.len(),main_tokens,total);
 Ok(())
}
