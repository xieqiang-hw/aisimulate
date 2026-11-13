"use strict";
window.AisimBsZoom=(()=>{
 const requests=new Map(),cache=new Map();
 window.aisimBsPackLoaded=(id,pack)=>{const request=requests.get(id);if(request){requests.delete(id);request.resolve(pack)}};
 async function decode(pack){
  if(typeof DecompressionStream!=="function")throw Error("当前浏览器不支持本地 gzip 解码，请使用新版 Chrome / Edge");
  const raw=Uint8Array.from(atob(pack.b64),c=>c.charCodeAt(0));
  const buffer=await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const bytes=new Uint8Array(buffer),view=new DataView(buffer);let offset=0;
  if(String.fromCharCode(...bytes.slice(0,4))!=="BSR1")throw Error("BS 数据包格式错误");
  offset=4;const streams=view.getUint32(offset,true);offset+=4;const result=[];
  function varint(){let value=0,factor=1;for(let i=0;i<8;i++){if(offset>=bytes.length)throw Error("BS 数据包不完整");const byte=bytes[offset++];value+=(byte&127)*factor;if(byte<128)return value;factor*=128;}throw Error("BS 时间编码越界")}
  for(let sid=0;sid<streams;sid++){
   const count=view.getUint32(offset,true)+view.getUint32(offset+4,true)*4294967296;offset+=8;
   if(count>20000000)throw Error("BS 计算段数量超出安全显示范围");
   const starts=new Float64Array(count),ends=new Float64Array(count),bs=new Uint32Array(count);
   let previous=0;
   for(let i=0;i<count;i++){const start=previous+varint(),end=start+varint(),batch=varint();starts[i]=start/1e6;ends[i]=end/1e6;bs[i]=batch;previous=end;}
   result.push({starts,ends,bs,count});
  }
  if(offset!==bytes.length)throw Error("BS 数据包含未解析数据");
  return result;
 }
 function load(id,descriptor){
  if(cache.has(id))return cache.get(id);
  const promise=new Promise((resolve,reject)=>{
   const script=document.createElement("script");let received=false;
   requests.set(id,{resolve:pack=>{received=true;decode(pack).then(resolve,reject)}});
   script.src=new URL(descriptor.url,location.href).href;
   script.onerror=()=>{requests.delete(id);reject(Error("精细 BS 文件未能加载；请保留 HTML 旁边的 analysis 数据目录"))};
   script.onload=()=>{if(!received){requests.delete(id);reject(Error("BS 数据包未返回预期实验"))}};
   document.head.append(script);
  });
  cache.set(id,promise);return promise;
 }
 function first(stream,time){let low=0,high=stream.count;while(low<high){const mid=(low+high)>>1;if(stream.ends[mid]<=time)low=mid+1;else high=mid;}return low}
 function segments(stream,from,to){
  const result=[];for(let i=first(stream,from);i<stream.count&&stream.starts[i]<to;i++){
   result.push({x:stream.starts[i],end:stream.ends[i],y:stream.bs[i]});
  }return result;
 }
 function step(span){
  const needed=span/600,steps=[1,2,5,10,15,30,60,120,300,600,900,1800,3600,7200,10800,21600,43200,86400,172800,604800];
  return steps.find(x=>x>=needed)||Math.ceil(needed/86400)*86400;
 }
 function aggregate(stream,from,to){
  const width=step(to-from),n=Math.max(1,Math.ceil((to-from)/width));
  const busy=new Float64Array(n),weighted=new Float64Array(n),peak=new Uint32Array(n);
  for(let i=first(stream,from);i<stream.count&&stream.starts[i]<to;i++){
   const a=Math.max(from,stream.starts[i]),b=Math.min(to,stream.ends[i]);
   const lo=Math.max(0,Math.floor((a-from)/width)),hi=Math.min(n-1,Math.floor((b-from)/width));
   for(let j=lo;j<=hi;j++){const duration=Math.min(b,from+(j+1)*width)-Math.max(a,from+j*width);if(duration>0){busy[j]+=duration;weighted[j]+=duration*stream.bs[i];peak[j]=Math.max(peak[j],stream.bs[i]);}}
  }
  return {step:width,points:Array.from({length:n},(_,i)=>{const x=from+i*width,end=Math.min(to,x+width);return {x,end,y:100*busy[i]/(end-x),peak:peak[i],mean:busy[i]?weighted[i]/busy[i]:0}})};
 }
 return {load,segments,aggregate};
})();