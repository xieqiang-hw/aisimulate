"use strict";
(() => {
 const source=meta.timelines||{catalog:[],traces:{}}, palette=["#19745c","#5279ba","#bb8540","#9b6eaa","#778b52","#ba6a5d"];
 const ids=["tlPChart","tlDChart","tlAuxChart"];
 let selected=null,windowData=null,from=0,to=1,base=0,unit=1,exportSeries=[],bsLevel='busy';
 const labelStatus={completed:"汇总中",running:"运行中",pending:"待启动",failed:"失败",interrupted:"已中断"};
 function pools(){return selected?.overview||[]}
 function pool(role){return pools().find(x=>x.identity.role===role)}
 function sid(role){return pool(role)?.stream_id}
 function selectCase(){
  selected=source.traces[$("tlCase").value]||null;
  const optionsList=[["overview","全程 / 自选范围 · 自动精度"],...(selected?.focus||[]).map(f=>[f.id,f.label+" · 逐轮与完整辅助"])];
  options($("tlWindow"),optionsList,"overview");
  if(selected?.full_bs&&!selected._fullBS&&!selected._fullLoading){
   const target=selected;target._fullLoading=true;
   AisimBsZoom.load($("tlCase").value,target.full_bs).then(streams=>{target._fullBS=streams;target._fullLoading=false;if(selected===target)renderTimeline()})
    .catch(error=>{target._fullLoading=false;target._fullError=error.message;if(selected===target)renderTimeline()});
  }
  resetWindow();
 }
 function resetWindow(){
  windowData=selected?.focus?.find(f=>f.id===$("tlWindow").value)||null;
  base=windowData?windowData.start_s:0;unit=windowData?1:86400;
  const end=windowData?(windowData.end_s-base):Math.max(1,(selected?.end_s||86400)/unit);
  $("tlFrom").value="0";$("tlTo").value=windowData?String(end):end.toFixed(9);
  for(const id of ["tlFrom","tlTo"]){$(id).step=windowData?"0.001":"any";$(id).max=String(end);$(id).min="0";}
  $("tlUnitA").textContent=$("tlUnitB").textContent=windowData?"（秒）":"（回放天）";
  renderTimeline();
 }
 function xLabel(x){return windowData?fmt(x-base,3)+" 秒":"第 "+fmt(x/86400,6)+" 天"+(to-from<=7200?" · 窗口内 "+fmt(x-from,3)+" 秒":"")}
 function point(x,y,extra={}){return {x,y,...extra}}
 function overviewSeries(role,column,name,color,missingZero=true){
  const p=pool(role);if(!p)return {name,color,points:[]};
  const map=new Map(p.rows.map(r=>[r[0],r])),max=Math.floor((selected.end_s||0)/3600);
  const points=[];for(let i=0;i<=max;i++){const r=map.get(i);points.push(point((i+.5)*3600,r?(missingZero||r[20]>0?r[column]:null):(missingZero?0:null),{note:"小时窗口 "+i+"～"+(i+1)}))}
  return {name,color,points};
 }
 function rawSeries(role,column,name,color,{duration=false}={}){
  const id=sid(role),rows=id===undefined?[]:(windowData?.points[id]||[]);
  return {name,color,segments:!duration,scatter:duration,points:rows.map(r=>point(r[2],duration?(r[3]-r[2])*1000:r[column],{end:r[3],iteration:r[0],note:"迭代 "+r[0]+" · "+xLabel(r[2])+"～"+xLabel(r[3])}))};
 }
 function stateSeries(column,index,name,color){
  const role=$("tlRole").value,p=pool(role),id=p?.stream_id;
  if(!p)return {name,color,points:[]};
  if(!windowData)return overviewSeries(role,index,name,color,false);
  return {name,color,scatter:true,points:(windowData.states[id]||[]).map(r=>point(r[1],r[column],{iteration:r[0],note:"调度采样 · 迭代 "+r[0]}))};
 }
 function plotTimeline(canvas,series,yunit,{percent=false,autoPercent=false,emptyText=""}={}){
  const {c,w,h}=context(canvas),rightAxis=series.some(s=>s.rightAxis),L=62,R=rightAxis?55:20,T=26,B=40,W=Math.max(1,w-L-R),H=Math.max(1,h-T-B);
  const visible=series.flatMap(s=>s.points.filter(p=>Number.isFinite(p.y)&&p.x<=to&&(p.end??p.x)>=from).map(p=>({...p,rightAxis:s.rightAxis})));
  const leftValues=visible.filter(p=>!p.rightAxis).map(p=>p.y),maxValue=Math.max(0,...leftValues);
  const ymax=(autoPercent?Math.max(.0001,maxValue):Math.max(1,maxValue))*1.12,isCount=["请求数","请求／任务数","次数"].includes(yunit);
  let top=percent?100:ymax,steps=4;
  if(maxValue===0&&!percent)top=1;
  if(isCount){top=Math.ceil(ymax);steps=top<=6?top:4;top=Math.ceil(top/steps)*steps;}
  const rightMax=Math.max(1,...visible.filter(p=>p.rightAxis).map(p=>p.y)),rightTop=Math.max(4,Math.ceil(rightMax*1.1/4)*4);
  const X=x=>L+(x-from)/(to-from)*W,Y=(y,right=false)=>T+H-y/(right?rightTop:top)*H;
  c.textAlign="left";c.fillText(yunit,L,13);
  for(let i=0;i<=steps;i++){const v=top*i/steps,y=Y(v);c.strokeStyle="#edf0eb";c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.textAlign="right";c.fillStyle="#61736c";c.fillText(v>=1e6?fmt(v/1e6,1)+"M":v.toLocaleString("zh-CN",{maximumFractionDigits:isCount?0:top<.1?3:top<5?2:top<50?1:0}),L-8,y+4);}
  if(rightAxis){c.textAlign="left";c.fillText("BS",w-R+7,13);for(let i=0;i<=4;i++){const v=rightTop*i/4;c.fillText(fmt(v,rightTop<4?1:0),w-R+7,Y(v,true)+4);}}
  const seconds=!!windowData||to-from<=7200,axisBase=windowData?base:(seconds||to-from<=172800?from:0),axisUnit=seconds?1:(to-from<=172800?3600:86400);
  c.textAlign="center";for(let i=0;i<=5;i++){const x=from+(to-from)*i/5;c.fillText(((x-axisBase)/axisUnit).toLocaleString("zh-CN",{maximumFractionDigits:2}),X(x),h-B+18);}
  c.textAlign="right";c.fillText(seconds?"窗口内秒":axisUnit===3600?"窗口内小时":"回放天数",w-R,h-4);
  c.save();c.beginPath();c.rect(L,T,W,H);c.clip();
  for(const seriesItem of series){
   c.strokeStyle=seriesItem.color;c.fillStyle=seriesItem.color;c.lineWidth=seriesItem.dashed?1.2:1.5;c.setLineDash(seriesItem.dashed?[4,4]:[]);
   let started=false;
   for(const point of seriesItem.points){
    if(!Number.isFinite(point.y)){if(started)c.stroke();started=false;continue;}
    if(point.x>to||(point.end??point.x)<from)continue;
    const x=X(point.x),y=Y(point.y,seriesItem.rightAxis);
    if(seriesItem.segments){
     const end=X(point.end??point.x),left=Math.max(L,x),right=Math.min(w-R,end);
     if(point.y>0){c.globalAlpha=.15;c.fillRect(left,y,Math.max(.7,right-left),T+H-y);c.globalAlpha=1;c.beginPath();c.moveTo(left,y);c.lineTo(Math.max(left+.8,right),y);c.stroke();}
    }else if(seriesItem.scatter){c.beginPath();c.arc(x,y,2.1,0,Math.PI*2);c.fill();}
    else{if(!started){c.beginPath();c.moveTo(x,y);started=true}else c.lineTo(x,y);}
    const px=seriesItem.segments?X((Math.max(from,point.x)+Math.min(to,point.end??point.x))/2):x;
    hover.get(canvas).push({x:Math.min(w-R,Math.max(L,px)),y,text:seriesItem.name+"\n"+xLabel(point.x)+"\n"+fmt(point.y,3)+" "+(seriesItem.unit||yunit)+(point.note?"\n"+point.note:"")});
   }
   if(started)c.stroke();
  }
  c.restore();c.setLineDash([]);
  if(!visible.length){c.fillStyle="#61736c";c.textAlign="center";c.fillText(emptyText||"此窗口没有对应阶段的计算／采样记录",w/2,h/2);}
  canvas.dataset.points=String(visible.length);canvas.dataset.axisMin="0";canvas.dataset.axisMax=String(top);canvas.dataset.mode=bsLevel;canvas.dataset.xUnit=seconds?"seconds":axisUnit===3600?"hours":"days";
 }
 function fullStream(role){return selected?._fullBS?.[sid(role)]}
 function aggregate(role){
  const stream=fullStream(role),key=from+":"+to;
  if(stream){
   selected._aggregate=selected._aggregate||{};
   if(selected._aggregate[role]?.key===key)return selected._aggregate[role].value;
   const value=AisimBsZoom.aggregate(stream,from,to);selected._aggregate[role]={key,value};return value;
  }
  const p=pool(role);return {step:3600,points:(p?.rows||[]).map(r=>({x:r[0]*3600,end:(r[0]+1)*3600,y:r[3],peak:r[1],mean:r[2]}))};
 }
 function intervalLabel(seconds){return seconds<60?fmt(seconds,seconds<1?3:seconds<10?2:1)+" 秒":seconds<3600?fmt(seconds/60,Number.isInteger(seconds/60)?0:1)+" 分钟":seconds<86400?fmt(seconds/3600,1)+" 小时":fmt(seconds/86400,1)+" 天"}
 function busySeries(role){
  const value=aggregate(role),color=role==="prefill"?palette[0]:palette[1];
  return {name:"计算繁忙占比",color,segments:true,unit:"%",points:value.points.map(p=>({...p,note:"统计桶 "+intervalLabel(p.end-p.x)+" · BS 峰值 "+p.peak+" · 执行期间平均 BS "+fmt(p.mean,3)}))};
 }
 function bsSeries(role){
  const color=role==="prefill"?palette[0]:palette[1],name=role==="prefill"?"Prefill":"Decode";
  if(bsLevel==="raw"){
   if(windowData)return [rawSeries(role,role==="prefill"?4:5,name+" 实际 BS",color)];
   return [{name:name+" 实际 BS",color,segments:true,points:AisimBsZoom.segments(fullStream(role),from,to).map(p=>({...p,note:"计算段 "+xLabel(p.x)+"～"+xLabel(p.end)+"；相邻且 BS 相同的段已合并"}))}];
  }
  const list=[busySeries(role)];
  if($("tlShowPeak").checked)list.push({name:"桶内 BS 峰值（右轴）",color:palette[2],rightAxis:true,unit:"BS",dashed:true,points:aggregate(role).points.map(p=>({x:(p.x+p.end)/2,y:p.peak}))});
  return list;
 }
 function applyRange(a,b){
  const minimum=windowData?windowData.start_s:0,maximum=windowData?windowData.end_s:selected.end_s;
  const minSpan=!windowData&&!selected._fullBS?Math.min(3600,maximum-minimum):.001;
  let width=Math.min(maximum-minimum,Math.max(minSpan,b-a)),center=(a+b)/2;
  center=Math.max(minimum+width/2,Math.min(maximum-width/2,center));
  $("tlFrom").value=((center-width/2-base)/unit).toFixed(9);$("tlTo").value=((center+width/2-base)/unit).toFixed(9);renderTimeline();
 }
 function scale(factor,center=(from+to)/2){const width=(to-from)*factor;applyRange(center-width/2,center+width/2)}
 function renderTimeline(){
  exportSeries=[];
  const count=Object.keys(source.traces).length;
  $("tlProgress").textContent="概览 "+count+"／"+source.catalog.length+" · 精细 "+source.catalog.filter(c=>c.full_bs_ready).length+"／"+source.catalog.length;
  if(!selected){
   $("tlExplain").textContent="本轮尚无已汇总时间线。后台会为已完成实验逐组准备图表；刷新页面读取最新版本。";
   for(const id of ids)plotTimeline($(id),[],"");return;
  }
  const a=Number($("tlFrom").value),b=Number($("tlTo").value);
  if(!Number.isFinite(a)||!Number.isFinite(b)||b<=a){$("tlExplain").textContent="时间终点需要大于起点。";return;}
  from=base+a*unit;to=base+b*unit;
  bsLevel=windowData||(selected._fullBS&&to-from<=300)?"raw":"busy";
  $("tlShowPeak").disabled=$("tlFixedBusyAxis").disabled=bsLevel==="raw";
  const precision=bsLevel==="raw"?"真实 BS 计算段":intervalLabel(aggregate("prefill").step)+" 聚合";
  $("tlScaleStatus").textContent="跨度 "+intervalLabel(to-from)+" · "+precision;
  $("tlExplain").textContent=windowData?
   "当前为"+windowData.label+"的原始逐轮记录；窗口起点在回放第 "+fmt(base/86400,4)+" 天。P/D 共用这段 60 秒，空闲区间不延续 BS。"+(windowData.truncated?" 此窗口超过展示上限，仅显示前 50,000 条，完整数据仍保存在原始文件中。":""):
   (bsLevel==="raw"?"当前为任意时间段的实际 BS：只填充真实计算区间，空闲留白；相邻且 BS 相同的计算段已合并（微秒级显示精度）。":
   "当前显示计算繁忙占比：面积高度表示桶内有多少时间在计算，不是 BS，也不代表计算发生在桶内的某一固定位置。拖选放大后桶会细化，缩到 5 分钟以内显示真实 BS。");
  $("tlSource").textContent=selected._fullBS?"全程精细 BS 已加载 · 拖选放大 / 双击缩小":selected._fullError|| (selected._fullLoading?"正在加载全程精细 BS…":"精细 BS 数据包准备中，暂用小时概览；预置分钟仍可查看");
  const p=pool("prefill"),d=pool("decode"),iterations=pools().reduce((n,p)=>n+p.iterations,0),pre=pools().flatMap(p=>p.rows).reduce((n,r)=>n+r[17],0),read=pools().flatMap(p=>p.rows).reduce((n,r)=>n+r[18],0);
  $("tlCards").replaceChildren(...[["P 峰值 BS",p?.max_bs??0,"全程实际执行峰值"],["D 峰值 BS",d?.max_bs??0,"全程实际执行峰值"],["记录迭代数",iterations,"包含零工作量重试"],["抢占 / 重新准入",pre+" / "+read,"两个阶段全程合计"]].map(([title,value,note])=>{const el=document.createElement("div");el.className="card";const label=document.createElement("label"),strong=document.createElement("strong"),small=document.createElement("p");label.textContent=title;strong.textContent=typeof value==="number"?value.toLocaleString("zh-CN"):value;small.textContent=note;el.append(label,strong,small);return el}));
  for(const [role,id,lid,subid] of [["prefill","tlPChart","tlPLegend","tlPSub"],["decode","tlDChart","tlDLegend","tlDSub"]]){
   const series=bsSeries(role);makeLegend($(lid),series.map(s=>[s.name,s.color]));
   const raw=bsLevel==="raw";$(id).closest(".chart").querySelector("h3").textContent=(role==="prefill"?"Prefill":"Decode")+(raw?" 实际 BS · 计算段":" 计算繁忙占比 · 可缩放");
   plotTimeline($(id),series,raw?"请求数":"%",{percent:!raw&&$("tlFixedBusyAxis").checked,autoPercent:!raw,emptyText:raw?"此窗口没有计算执行，BS 为 0":""});
   $(subid).textContent=(raw?(windowData?"原始逐轮；空闲留白":"等 BS 连续段合并；空闲留白"):"每 "+intervalLabel(aggregate(role).step)+" 聚合；100% 表示桶内持续计算")+($("tlShowPeak").checked&&!raw?" · 右轴为 BS 峰值":"")+" · max_sequences="+(pool(role)?.identity.max_sequences??"—");
   exportSeries.push(...series);
  }
  const key=$("tlMetric").value,role=$("tlRole").value,roleName=role==="prefill"?"P":"D";
  let series=[],title="",subtitle="",units="",percent=false;
  if(key==="prefill"){title="Prefill 每轮新计算 token 数";units="token";series=[windowData?rawSeries("prefill",6,"新计算 token",palette[0]):overviewSeries("prefill",6,"每轮 token 数的小时最大值",palette[0])];subtitle="缓存命中和此前 chunk 的 token 不计入新计算量。"}
  if(key==="duration"){title="Decode 每轮计算耗时";units="ms";series=[windowData?rawSeries("decode",0,"单轮计算耗时",palette[1],{duration:true}):overviewSeries("decode",8,"单轮耗时的小时最大值",palette[1])];subtitle="end − compute_start；不等于单条请求 ITL，也不包含轮前等待。"}
  if(key==="context"){title="Decode 已有 KV 上下文规模";units="token";series=windowData?[rawSeries("decode",7,"本轮 KV token 总和",palette[1]),rawSeries("decode",8,"单请求 KV 长度最大值",palette[2])]:[overviewSeries("decode",7,"每轮 KV 总量的小时最大值",palette[1])];subtitle="同样的 BS 也可能对应不同 KV 访问量。"}
  if(key==="queue"){title=roleName+" 侧等待队列与运行请求";units="请求数";series=[stateSeries(2,9,"普通准入等待",palette[0]),stateSeries(3,10,"连接器等待",palette[2]),stateSeries(7,14,"运行集合",palette[1])];subtitle="调度采样；普通等待不等于 KV 已就绪。"+(windowData?"短窗为状态变化点。":"概览为采样最大值。")}
  if(key==="restore"){title=roleName+" 侧缓存恢复与 SSD 任务";units="请求／任务数";series=[stateSeries(4,11,"H2D 加载请求",palette[0]),stateSeries(5,12,"SSD 在途读任务",palette[1]),stateSeries(6,13,"SSD 在途写任务",palette[2])];subtitle=(role==="decode"?"D 本地 G3 未启用；SSD 曲线不适用，H2D 加载仍可发生。":"H2D 请求数与 SSD 任务数为不同计数单位，不相加。");if(role==="decode")series=series.slice(0,1)}
  if(key==="hbm"){title=roleName+" 侧 HBM KV 占用";units="%";percent=true;const cap=pool(role)?.identity.hbm_total_blocks||1;series=[stateSeries(8,15,"被引用块 / 容量",palette[0]),stateSeries(9,16,"总驻留块 / 容量",palette[1])].map(s=>({...s,points:s.points.map(p=>({...p,y:p.y===null?null:100*p.y/cap}))}));subtitle="总驻留包含可复用的未引用块；采样发生在调度提交后、输出生成前。"}
  if(key==="busy"){title="计算繁忙时间占比";units="%";percent=true;if(!windowData)series=["prefill","decode"].map((r,i)=>({...busySeries(r),name:(r==="prefill"?"P":"D")+" 计算繁忙占比",color:palette[i]}));else{series=["prefill","decode"].map((r,i)=>{const rows=windowData.points[sid(r)]||[];const bins=Array(60).fill(0);for(const row of rows){let start=Math.max(base,row[2]),end=Math.min(base+60,row[3]);while(start<end){const index=Math.floor(start-base);if(index<0||index>=60)break;const stop=Math.min(end,base+index+1);bins[index]+=stop-start;start=stop;}}return {name:(r==="prefill"?"P":"D")+" 计算时长 / 秒",color:palette[i],points:bins.map((v,i)=>point(base+i+.5,v*100))}})}subtitle="按计算时间段与窗口的重叠时长计算；不是硬件实测的 GPU 利用率。"}
  if(key==="pressure"){title="抢占与重新准入";units="次数";if(!windowData)series=[overviewSeries(role,17,"每小时抢占次数",palette[2]),overviewSeries(role,18,"每小时重新准入次数",palette[1])];else series=[{name:"本轮抢占数",color:palette[2],scatter:true,points:(windowData.points[sid(role)]||[]).filter(r=>r[10]>0).map(r=>point(r[1],r[10],{iteration:r[0]}))}];subtitle=windowData?"本轮抢占数；完整重新准入事件保存在 scheduling_events 文件，小时概览可查看其计数。":"重新准入指同一调度器内被抢占后重新进入，不是常规 P→D 交接。"}
  if(!windowData&&to-from<3600&&key!=="busy"){series=[];subtitle+=" 当前范围小于辅助指标的一小时统计精度；请选预置最忙分钟查看其逐轮辅助细节。BS 图使用全程精细计算段，不受此限制。";}
  $("tlAuxTitle").textContent=title;$("tlAuxSub").textContent=subtitle;
  $("tlRole").disabled=!["queue","restore","hbm","pressure"].includes(key);
  makeLegend($("tlAuxLegend"),series.map(s=>[s.name,s.color]));plotTimeline($("tlAuxChart"),series,units,{percent:key==="busy"?$("tlFixedBusyAxis").checked:percent,autoPercent:key==="busy"});
  exportSeries.push(...series);
 }
 const catalog=source.catalog||[];
 $("tlCase").replaceChildren(...catalog.map(row=>{const option=new Option(({"baseline":"基线","hbm":"HBM","overlay":"叠加","bandwidth":"带宽"})[row.suite]+" · "+row.label+(row.timeline_ready?(row.full_bs_ready?"":" · 缩放包准备中"):" · "+(labelStatus[row.status]||row.status)),row.id);option.disabled=!row.timeline_ready;return option}));
 const ready=catalog.find(row=>row.timeline_ready);if(ready)$("tlCase").value=ready.id;
 $("tlCase").addEventListener("change",selectCase);$("tlWindow").addEventListener("change",resetWindow);
 ["tlFrom","tlTo","tlMetric","tlRole"].forEach(id=>$(id).addEventListener("change",renderTimeline));
 $("tlReset").addEventListener("click",resetWindow);
 $("tlWhole").addEventListener("click",()=>{$("tlWindow").value="overview";resetWindow()});
 $("tlZoomIn").addEventListener("click",()=>{if(selected)scale(.25)});
 $("tlZoomOut").addEventListener("click",()=>{if(selected)scale(4)});
 ["tlShowPeak","tlFixedBusyAxis"].forEach(id=>$(id).addEventListener("change",renderTimeline));
 for(const id of ["tlPChart","tlDChart"]){
  const canvas=$(id),brush=document.createElement("div");brush.className="tl-brush";canvas.parentElement.append(brush);
  let dragging=null;
  function position(event){const box=canvas.getBoundingClientRect(),right=bsLevel==="busy"&&$("tlShowPeak").checked?55:20;return Math.max(62,Math.min(box.width-right,event.clientX-box.left))}
  canvas.addEventListener("pointerdown",event=>{if(event.button!==0||!selected)return;dragging={x:position(event),from,to};canvas.setPointerCapture(event.pointerId);brush.style.left=dragging.x+"px";brush.style.width="0px";brush.style.display="block";});
  canvas.addEventListener("pointermove",event=>{if(!dragging)return;const x=position(event);brush.style.left=Math.min(x,dragging.x)+"px";brush.style.width=Math.abs(x-dragging.x)+"px";});
  canvas.addEventListener("pointerup",event=>{if(!dragging)return;const x=position(event),start=dragging;dragging=null;brush.style.display="none";if(Math.abs(x-start.x)<8)return;const right=bsLevel==="busy"&&$("tlShowPeak").checked?55:20,width=canvas.getBoundingClientRect().width-62-right;const time=p=>start.from+(p-62)/width*(start.to-start.from);applyRange(time(Math.min(x,start.x)),time(Math.max(x,start.x)));});
  canvas.addEventListener("pointercancel",()=>{dragging=null;brush.style.display="none";});
  canvas.addEventListener("dblclick",()=>{if(selected)scale(4)});
 }

 $("tlExport").addEventListener("click",()=>{
  const rows=[["曲线","仿真开始秒","仿真结束秒","数值","迭代编号"]];
  for(const s of exportSeries)for(const p of s.points)if(Number.isFinite(p.y)&&p.x<=to&&(p.end??p.x)>=from)rows.push([s.name,p.x,p.end??p.x,p.y,p.iteration??""]);
  const csv="\ufeff"+rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"})),a=document.createElement("a");a.href=url;a.download="scheduler-timeline.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 let timer;window.addEventListener("resize",()=>{clearTimeout(timer);timer=setTimeout(renderTimeline,150)});
 selectCase();
})();