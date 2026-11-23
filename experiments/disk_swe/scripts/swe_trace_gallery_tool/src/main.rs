// SPDX-License-Identifier: Apache-2.0
//! Offline index of a paired SWE-chat Mooncake bundle; never runs a simulation.
use std::{collections::{HashMap,HashSet},fs::File,io::{BufRead,BufReader,BufWriter},path::Path};
use serde::{Deserialize,Serialize};
use serde_json::json;
#[derive(Deserialize)]struct Trace{timestamp:f64,input_length:usize,output_length:usize,hash_ids:Vec<u64>}
#[derive(Deserialize)]struct Source{session_id:String,call_index:usize,transcript_path:String,model:Option<String>,trace_row:usize,timestamp:f64,#[serde(default)]history_reset:bool,first_response_event_at:String}
#[derive(Serialize)]struct Session{id:String,path:String,model:Option<String>,first_response_at:String,first_ms:f64,last_ms:f64,requests:usize,max_input:usize,#[serde(skip_serializing)]hashes:Vec<u64>,#[serde(skip_serializing)]previous_row:Option<usize>,#[serde(skip_serializing)]last_call:usize,#[serde(skip_serializing)]last_input:usize}
type Row=(usize,f64,usize,usize,usize,Option<usize>,Option<usize>,bool,usize,Option<usize>);
fn main()->anyhow::Result<()>{
 let a:Vec<_>=std::env::args().collect();anyhow::ensure!(a.len()>=4,"usage: swe-trace-gallery index BUNDLE OUT | extract BUNDLE ROWS_JSON OUT");let base=Path::new(&a[2]);
 if a[1]=="extract"{anyhow::ensure!(a.len()==5,"extract needs output path");let selected:Vec<usize>=serde_json::from_reader(File::open(&a[3])?)?;let set:HashSet<_>=selected.iter().copied().collect();let mut out=Vec::new();let mut tr=BufReader::with_capacity(1048576,File::open(base.join("requests.jsonl"))?);let mut sr=BufReader::with_capacity(262144,File::open(base.join("source_requests.jsonl"))?);let mut t=String::new();let mut s=String::new();let mut i=0;loop{t.clear();s.clear();if tr.read_line(&mut t)?==0{break}anyhow::ensure!(sr.read_line(&mut s)?>0,"source ended early");if set.contains(&i){out.push(json!({"trace_row":i,"request":serde_json::from_str::<serde_json::Value>(&t)?,"source":serde_json::from_str::<serde_json::Value>(&s)?}));}i+=1;}anyhow::ensure!(out.len()==set.len(),"selected rows missing");serde_json::to_writer(BufWriter::new(File::create(&a[4])?),&out)?;println!("Extracted {} rows",out.len());return Ok(())}
 anyhow::ensure!(a[1]=="index","unknown command");
 let mut tr=BufReader::with_capacity(1048576,File::open(base.join("requests.jsonl"))?);let mut sr=BufReader::with_capacity(262144,File::open(base.join("source_requests.jsonl"))?);let mut t=String::new();let mut s=String::new();let mut sessions:Vec<Session>=Vec::new();let mut ids=HashMap::new();let mut rows:Vec<Row>=Vec::new();let mut nonmonotone=0;let mut gaps=0;let mut previous_stamp=0.0;
 loop{t.clear();s.clear();if tr.read_line(&mut t)?==0{break}anyhow::ensure!(sr.read_line(&mut s)?>0,"source ended early");let q:Trace=serde_json::from_str(&t)?;let src:Source=serde_json::from_str(&s)?;let row=rows.len();anyhow::ensure!(src.trace_row==row&&(src.timestamp-q.timestamp).abs()<0.001,"source row mismatch");anyhow::ensure!(q.timestamp>=previous_stamp,"unsorted trace");previous_stamp=q.timestamp;anyhow::ensure!(q.hash_ids.len()==q.input_length.div_ceil(16),"hash count mismatch");
  let sid=if let Some(&i)=ids.get(&src.session_id){i}else{let i=sessions.len();ids.insert(src.session_id.clone(),i);sessions.push(Session{id:src.session_id,path:src.transcript_path,model:src.model,first_response_at:src.first_response_event_at,first_ms:q.timestamp,last_ms:q.timestamp,requests:0,max_input:0,hashes:Vec::new(),previous_row:None,last_call:0,last_input:0});i};
  let session=&mut sessions[sid];let prev=session.previous_row;let lcp=prev.map(|_|session.hashes.iter().zip(&q.hash_ids).take_while(|(a,b)|a==b).count()*16).map(|v|v.min(q.input_length).min(session.last_input));
  if prev.is_some(){if src.call_index<=session.last_call{nonmonotone+=1;}if src.call_index!=session.last_call+1{gaps+=1;}}
  rows.push((sid,q.timestamp,q.input_length,q.output_length,src.call_index,lcp,lcp.map(|v|v/64*64),src.history_reset,row,prev));session.requests+=1;session.last_ms=q.timestamp;session.max_input=session.max_input.max(q.input_length);session.hashes=q.hash_ids;session.previous_row=Some(row);session.last_call=src.call_index;session.last_input=q.input_length;
  if rows.len()%100000==0{println!("Indexed {} requests / {} sessions",rows.len(),sessions.len())}
 }
 s.clear();anyhow::ensure!(sr.read_line(&mut s)?==0,"extra source rows");
 serde_json::to_writer(BufWriter::new(File::create(&a[3])?),&json!({"version":1,"columns":["session","timestamp_ms","input_tokens","output_tokens","call_index","previous_prefix16_tokens","previous_prefix64_tokens","history_reset","trace_row","previous_row"],"sessions":sessions,"rows":rows,"nonmonotone_call_pairs":nonmonotone,"nonadjacent_call_pairs":gaps}))?;
 println!("Saved {} requests / {} sessions",rows.len(),sessions.len());Ok(())
}
