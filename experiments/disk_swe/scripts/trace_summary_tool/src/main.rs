// SPDX-License-Identifier: Apache-2.0
use std::{fs::File, io::{BufRead, BufReader, Write}, path::Path};
use flate2::read::GzDecoder;
use serde::Deserialize;
use serde_json::{Value, json};

type Row = (usize,u64,f64,f64,f64,usize,usize,usize,usize,usize,usize,usize,usize);
const HOUR: f64 = 3_600_000.0;
const MINUTE: f64 = 60_000.0;
#[derive(Clone,Default)]
struct Bin {
    count:u64, peak:usize, busy:f64, bs_time:f64, prefill:u64, generated:u64,
    max_prefill:usize, max_kv:usize, max_duration:f64, preemptions:u64,
    queue:[usize;6], active:usize, resident:usize, readmissions:u64, samples:u64,
}
#[derive(Clone,Default,Deserialize)]
struct State {
    #[serde(default)] waiting_admission_requests:usize,
    #[serde(default)] connector_waiting_requests:usize,
    #[serde(default)] h2d_loading_requests:usize,
    g3_pending_read_jobs:Option<usize>,
    g3_pending_write_jobs:Option<usize>,
    #[serde(default)] running_requests:usize,
    #[serde(default)] hbm_active_blocks:usize,
    #[serde(default)] hbm_inactive_blocks:usize,
}
#[derive(Deserialize)]
struct Event { stream_id:usize,event:String,at_ms:f64 }
struct Pool { bins:Vec<Bin>, minute_busy:Vec<f64>, total:u64,prefill:u64,generated:u64,max_bs:usize }
impl Pool {
 fn new()->Self{Self{bins:vec![],minute_busy:vec![],total:0,prefill:0,generated:0,max_bs:0}}
 fn bin(&mut self,i:usize)->&mut Bin{
  if self.bins.len()<=i{self.bins.resize_with(i+1,Bin::default);}
  &mut self.bins[i]
 }
}
fn reader(dir:&Path,name:&str)->anyhow::Result<Box<dyn BufRead>>{
 let compressed=dir.join(format!("{name}.jsonl.gz"));
 Ok(if compressed.exists(){Box::new(BufReader::with_capacity(262144,GzDecoder::new(File::open(compressed)?)))}
 else{Box::new(BufReader::with_capacity(262144,File::open(dir.join(format!("{name}.jsonl")))?))})
}
fn scan<T:for<'a>Deserialize<'a>>(dir:&Path,name:&str,mut f:impl FnMut(T))->anyhow::Result<()>{
 let mut r=reader(dir,name)?;let mut line=String::with_capacity(1024);
 loop{line.clear();if r.read_line(&mut line)?==0{break} f(serde_json::from_str(&line)?);}
 Ok(())
}
fn next_state(reader:&mut dyn BufRead,line:&mut String)->anyhow::Result<Option<(usize,u64,f64,State)>>{
 line.clear();if reader.read_line(line)?==0{return Ok(None)}
 Ok(Some(serde_json::from_str(line)?))
}
fn sample(b:&mut Bin,s:&State){
 let values=[s.waiting_admission_requests,s.connector_waiting_requests,s.h2d_loading_requests,
             s.g3_pending_read_jobs.unwrap_or(0),s.g3_pending_write_jobs.unwrap_or(0),s.running_requests];
 for i in 0..6{b.queue[i]=b.queue[i].max(values[i]);}
 b.active=b.active.max(s.hbm_active_blocks);b.resident=b.resident.max(s.hbm_active_blocks+s.hbm_inactive_blocks);
 b.samples+=1;
}
fn round(v:f64)->f64{(v*1e6).round()/1e6}
fn varint(writer:&mut impl Write,mut value:u64)->anyhow::Result<()>{
 while value>=128 {writer.write_all(&[((value as u8)&127)|128])?;value>>=7;}
 writer.write_all(&[value as u8])?;Ok(())
}
fn segments(dir:&Path,out:&Path)->anyhow::Result<()>{
 let meta:Value=serde_json::from_reader(File::open(dir.join("scheduler_trace_metadata.json"))?)?;
 let identities=meta["streams"].as_array().unwrap();
 let mut runs:Vec<Vec<(u64,u64,u64)>>=identities.iter().map(|_|Vec::new()).collect();
 scan::<Row>(dir,"iteration_metrics",|r|{
  let bs=if identities[r.0]["role"]=="prefill"{r.5}else{r.8};
  let mut start=(r.3*1000.0).round()as u64;let end=(r.4*1000.0).round()as u64;
  if bs==0||end<=start{return}
  let list=&mut runs[r.0];
  if let Some(last)=list.last_mut(){
   if start<last.1{assert!(last.1-start<=1,"overlapping compute spans require a different BS representation");start=last.1;}
   if start==last.1&&last.2==bs as u64{last.1=end;return}
  }
  if end>start{list.push((start,end,bs as u64));}
 })?;
 let tmp=out.with_extension("tmp");
 let mut writer=std::io::BufWriter::with_capacity(262144,File::create(&tmp)?);
 writer.write_all(b"BSR1")?;writer.write_all(&(runs.len()as u32).to_le_bytes())?;
 for list in &runs{
  writer.write_all(&(list.len()as u64).to_le_bytes())?;let mut previous=0u64;
  for &(start,end,bs) in list{varint(&mut writer,start-previous)?;varint(&mut writer,end-start)?;varint(&mut writer,bs)?;previous=end;}
 }
 writer.flush()?;std::fs::rename(tmp,out)?;
 println!("BS segments {:?} -> {}",runs.iter().map(|r|r.len()).collect::<Vec<_>>(),out.display());Ok(())
}
fn main()->anyhow::Result<()>{
 let args:Vec<_>=std::env::args().collect();
 if args.len()==4&&args[1]=="--segments"{return segments(Path::new(&args[2]),Path::new(&args[3]));}
 anyhow::ensure!(args.len()==3,"usage: summarize_trace INPUT_DIRECTORY OUTPUT_JSON");
 let dir=Path::new(&args[1]);
 let meta:Value=serde_json::from_reader(File::open(dir.join("scheduler_trace_metadata.json"))?)?;
 anyhow::ensure!(meta["schema_version"]==2,"requires compact schema v2");
 let identities=meta["streams"].as_array().unwrap();
 let mut pools:Vec<_>=identities.iter().map(|_|Pool::new()).collect();
 let mut end:f64=0.0;
 let mut state_reader=reader(dir,"scheduler_states")?;let mut state_line=String::new();
 let mut upcoming=next_state(&mut *state_reader,&mut state_line)?;
 let mut states=vec![State::default();identities.len()];let mut seen=vec![false;identities.len()];
 scan::<Row>(dir,"iteration_metrics",|r|{
  let p=&mut pools[r.0];let is_p=identities[r.0]["role"]=="prefill";
  let bs=if is_p{r.5}else{r.8};
  p.total+=1;p.prefill+=r.6 as u64;p.generated+=r.11 as u64;p.max_bs=p.max_bs.max(bs);
  if upcoming.as_ref().is_some_and(|v|v.0==r.0&&v.1==r.1){
   let change=upcoming.take().unwrap();states[r.0]=change.3;seen[r.0]=true;
   upcoming=next_state(&mut *state_reader,&mut state_line).expect("valid completed state stream");
  }
  assert!(seen[r.0],"every iteration must have a baseline state");
  sample(p.bin((r.2/HOUR)as usize),&states[r.0]);
  end=end.max(r.4);
  let b=p.bin((r.3/HOUR) as usize);
  b.count+=1;b.peak=b.peak.max(bs);b.prefill+=r.6 as u64;b.generated+=r.11 as u64;
  b.max_prefill=b.max_prefill.max(r.6);b.max_kv=b.max_kv.max(r.9);
  b.max_duration=b.max_duration.max(r.4-r.3);b.preemptions+=r.12 as u64;
  let mut start=r.3;
  while start<r.4 {
   let i=(start/HOUR)as usize;let stop=r.4.min((i+1)as f64*HOUR);let duration=stop-start;
   let b=p.bin(i);b.busy+=duration;b.bs_time+=duration*bs as f64;b.peak=b.peak.max(bs);
   start=stop;
  }
  let mut start=r.3;
  while start<r.4{
   let i=(start/MINUTE)as usize;let stop=r.4.min((i+1)as f64*MINUTE);
   if p.minute_busy.len()<=i{p.minute_busy.resize(i+1,0.0)}
   p.minute_busy[i]+=stop-start;start=stop;
  }
 })?;
 anyhow::ensure!(upcoming.is_none(),"unmatched state transition");
 scan::<Event>(dir,"scheduling_events",|e|{
  if e.event=="readmission"{pools[e.stream_id].bin((e.at_ms/HOUR)as usize).readmissions+=1;}
 })?;
 // One exact minute per stream: maximal modeled compute time, first tie wins.
 let mut focus=Vec::new();
 for (sid,p) in pools.iter().enumerate(){
  let mut best=0;let mut busy=0.0;
  for (i,&value) in p.minute_busy.iter().enumerate(){if value>busy+1e-6{best=i;busy=value;}}
  focus.push(json!({"id":format!("stream-{sid}"),"stream_id":sid,
    "label":format!("{} 最忙分钟",if identities[sid]["role"]=="prefill"{"P"}else{"D"}),
    "start_s":best as f64*60.0,"end_s":(best+1)as f64*60.0,
    "busy_ms":round(busy),"points":identities.iter().map(|_|Vec::<Value>::new()).collect::<Vec<_>>(),
    "states":identities.iter().map(|_|Vec::<Value>::new()).collect::<Vec<_>>(),
    "truncated":false}));
 }
 scan::<Row>(dir,"iteration_metrics",|r|{
  for f in &mut focus{
   let begin=f["start_s"].as_f64().unwrap()*1000.0;let finish=f["end_s"].as_f64().unwrap()*1000.0;
   if (r.3<finish&&r.4>begin)||(r.3>=begin&&r.3<finish&&r.3==r.4){
    let list=f["points"][r.0].as_array_mut().unwrap();
    if list.len()<50000 {
     list.push(json!([r.1,round(r.2/1000.0),round(r.3/1000.0),round(r.4/1000.0),
                     r.5,r.8,r.6,r.9,r.10,r.11,r.12]));
    }else{f["truncated"]=json!(true);}
   }
  }
 })?;
 scan::<(usize,u64,f64,State)>(dir,"scheduler_states",|(sid,iteration,at,s)|{
  for f in &mut focus{
   let begin=f["start_s"].as_f64().unwrap()*1000.0;let finish=f["end_s"].as_f64().unwrap()*1000.0;
   if at>=begin&&at<finish{
    let list=f["states"][sid].as_array_mut().unwrap();
    if list.len()<50000 {
     list.push(json!([iteration,round(at/1000.0),s.waiting_admission_requests,s.connector_waiting_requests,
       s.h2d_loading_requests,s.g3_pending_read_jobs.unwrap_or(0),s.g3_pending_write_jobs.unwrap_or(0),
       s.running_requests,s.hbm_active_blocks,s.hbm_active_blocks+s.hbm_inactive_blocks]));
    }else{f["truncated"]=json!(true);}
   }
  }
 })?;
 let overview:Vec<Value>=pools.iter().enumerate().map(|(sid,p)|{
  let rows:Vec<Value>=p.bins.iter().enumerate().filter(|(_,b)|b.count>0||b.busy>0.0||b.readmissions>0||b.samples>0).map(|(i,b)|
   json!([i, b.peak, round(if b.busy>0.0{b.bs_time/b.busy}else{0.0}),round(b.busy/HOUR*100.0),
     b.prefill,b.generated,b.max_prefill,b.max_kv,round(b.max_duration),
     b.queue[0],b.queue[1],b.queue[2],b.queue[3],b.queue[4],b.queue[5],b.active,b.resident,
     b.preemptions,b.readmissions,b.count,b.samples])).collect();
  json!({"identity":identities[sid],"stream_id":sid,"rows":rows,"iterations":p.total,
    "prefill_tokens":p.prefill,"decode_generated_tokens":p.generated,"max_bs":p.max_bs})
 }).collect();
 let output=json!({"schema_version":1,"source_schema":2,"end_s":round(end/1000.0),
  "overview_bin_seconds":3600,"overview_columns":["hour","max_bs","active_time_weighted_mean_bs","busy_percent",
   "prefill_tokens","decode_generated_tokens","max_prefill_tokens_per_iteration","max_decode_kv_sum",
   "max_compute_duration_ms","waiting_admission_max","connector_waiting_max","h2d_loading_max",
   "g3_read_jobs_max","g3_write_jobs_max","running_max","hbm_active_max","hbm_resident_max",
   "preemptions","readmissions","iterations","state_samples"],
  "raw_columns":["iteration_id","start_s","compute_start_s","end_s","prefill_bs","decode_bs",
    "prefill_new_tokens","decode_kv_sum","decode_kv_max","decode_generated_tokens","preemptions"],
  "state_columns":["iteration_id","at_s","waiting_admission","connector_waiting","h2d_loading",
    "g3_reads","g3_writes","running","hbm_active","hbm_resident"],
  "overview":overview,"focus":focus});
 let out=Path::new(&args[2]);let tmp=out.with_extension("tmp");
 let mut file=File::create(&tmp)?;serde_json::to_writer(&mut file,&output)?;file.flush()?;
 std::fs::rename(tmp,out)?;
 println!("{} streams, {} iterations -> {}",pools.len(),pools.iter().map(|p|p.total).sum::<u64>(),out.display());
 Ok(())
}
