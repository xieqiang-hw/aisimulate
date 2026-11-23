// Offline DOM/canvas logic checks. No browser, rendering or network access.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const page=fs.readFileSync(process.argv[2],'utf8');
const embedded=page.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=[...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m=>m[1]).at(-1);
const data=JSON.parse(embedded),elements=new Map();let csv,drawCalls=0;
const context2d=new Proxy({}, {get(target,key){if(key in target)return target[key];return (...args)=>{drawCalls++;for(const arg of args)if(typeof arg==='number')assert(Number.isFinite(arg),'nonfinite canvas argument in '+key)}},set(target,key,value){target[key]=value;return true}});
class Element{
 constructor(id=''){this.id=id;this.value='';this.textContent='';this.children=[];this.disabled=false;this.hidden=false;this.style={};this.classList={toggle(){}};this.parentElement={style:{},getBoundingClientRect:()=>({width:1000,height:500}),querySelector:()=>({style:{}})}}
 replaceChildren(...children){this.children=children;this.value=children[0]?String(children[0].value):''}
 getBoundingClientRect(){return {width:1000,height:305,left:0,top:0}}
 getContext(){return context2d}
 setAttribute(){}
 click(){if(this.onclick)this.onclick()}
}
for(const m of page.matchAll(/\bid="([^"]+)"/g))elements.set(m[1],new Element(m[1]));
elements.get('data').textContent=embedded;elements.get('axis').value='local';elements.get('comparisonPanel').hidden=true;
const context={document:{getElementById:id=>{assert(elements.has(id),'missing element '+id);return elements.get(id)},createElement:()=>new Element(),querySelectorAll:()=>[],body:{dataset:{}}},window:{addEventListener(){}},devicePixelRatio:1,location:{search:'?test'},Blob:class{constructor(parts){csv=parts.join('')}},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},setTimeout:callback=>{callback();return 1},clearTimeout(){}};
vm.createContext(context);vm.runInContext(code,context);
assert.equal(context.window.reportTest.rows,data.rows.length);assert(drawCalls>0);
function selected(){return vm.runInContext('selection',context)}
assert(selected().every(r=>r.dataset==='qwen-chat'));
elements.get('datasetSelect').value='mooncake-chat';elements.get('datasetSelect').onchange();
assert(selected().length>0);assert(selected().every(r=>r.dataset==='mooncake-chat'));
assert(elements.get('dataset').innerHTML.includes('mooncake-chat'));
elements.get('readTab').click();assert.equal(vm.runInContext('mode',context),'read');
elements.get('compareTab').click();assert.equal(vm.runInContext('mode',context),'compare');assert.equal(elements.get('mainCharts').hidden,true);assert.equal(elements.get('comparisonPanel').hidden,false);
elements.get('capacityTab').click();assert.equal(vm.runInContext('mode',context),'capacity');
elements.get('axis').value='zero';elements.get('axis').onchange();
elements.get('download').click();assert(csv.includes('mooncake-chat'));assert(!csv.includes('qwen-chat'));assert(csv.includes('P99 TTFT'));
assert.equal(vm.runInContext("value({status:'运行中',values:{sla:99}},'sla')",context),null);
const ds=Object.keys(data.datasets).find(name=>data.rows.some(r=>r.dataset===name));elements.get('datasetSelect').value=ds;elements.get('datasetSelect').onchange();
const available=[...new Set(selected().map(r=>r.read_gbps))];elements.get('bandwidth').value=String(available[0]);elements.get('bandwidth').onchange();assert(selected().every(r=>r.read_gbps===available[0]));
elements.get('bandwidth').value='all';elements.get('bandwidth').onchange();
for(const id of ['capacityTab','readTab','compareTab','stackChart','ssdReadChart','tableRows','scopeRows'])assert(elements.has(id));
console.log(JSON.stringify({checks:'passed',rows:data.rows.length,datasetIsolation:true,tabs:3,canvasCalls:drawCalls,csv:true,missingValues:true,bandwidthFilter:true}));
