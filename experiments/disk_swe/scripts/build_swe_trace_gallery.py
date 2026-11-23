"""Build a standalone SWE-chat trace gallery from the existing 128K base bundle.

Dataset attribution: SALT-NLP/SWE-chat, local download revision recorded in
analysis/swe_trace_gallery/provenance.json; ODC-By. See THIRD_PARTY_NOTICES.md.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

BASE=Path(__file__).resolve().parents[1]
FOLDER=BASE/'analysis/swe_trace_gallery'
BUNDLE=Path.home()/'project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k'
UPSTREAM=Path.home()/'project/data/swe_chat/upstream'
TOOL=Path.home()/'build/trace-summary/release/swe-trace-gallery'


def lcp(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i*16
    return min(len(a),len(b))*16


def content_text(content):
    if isinstance(content,str):return content
    if isinstance(content,list):
        return '\n'.join(content_text(v) for v in content)
    if isinstance(content,dict):
        if content.get('type') in ('text','tool_result'):return content_text(content.get('text',content.get('content','')))
        if content.get('type')=='tool_use':return '[工具调用：'+content.get('name','unknown')+']'
    return ''


def shorten(text,n=230):
    text=' '.join(text.split())
    return text[:n]+('…' if len(text)>n else '')


def main():
    FOLDER.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((BUNDLE/'manifest.json').read_text())
    for name in ['requests.jsonl','source_requests.jsonl']:
        assert (BUNDLE/name).stat().st_size==manifest['files'][name]['size']
    index=FOLDER/'request_index.json'
    if not index.exists():subprocess.run([str(TOOL),'index',str(BUNDLE),str(index)],check=True)
    raw=json.loads(index.read_text());rows=raw['rows'];sessions=raw['sessions'];groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[0]].append(i)
    assert len(rows)==manifest['request_count']
    t=np.array([r[1] for r in rows]);sid=np.array([r[0] for r in rows]);inputs=np.array([r[2] for r in rows]);outputs=np.array([r[3] for r in rows])
    assert int(inputs.sum())==manifest['diagnostics']['emitted_input_tokens']
    bins=(t//300000).astype(int);counts=np.bincount(bins);active=np.bincount(np.unique(bins*len(sessions)+sid)//len(sessions),minlength=len(counts))
    day=(t//86400000).astype(int);day_counts=np.bincount(day);day_active=np.bincount(np.unique(day*len(sessions)+sid)//len(sessions),minlength=len(day_counts))
    day_input=np.bincount(day,weights=inputs).astype(np.int64);day_output=np.bincount(day,weights=outputs).astype(np.int64)
    pair=[r for r in rows if r[5] is not None];ratio=np.array([r[5]/r[2] for r in pair]);ratio64=np.array([r[6]/r[2] for r in pair])
    paired_input=sum(r[2] for r in pair);matched=sum(r[5] for r in pair);matched64=sum(r[6] for r in pair)
    high=[];growth=[];drop=[]
    for s,ids in groups.items():
        rr=[rows[i] for i in ids];pp=[r for r in rr if r[5] is not None]
        if len(pp)<8:continue
        med=float(np.median([r[5]/r[2] for r in pp]));gaps=sum(rr[k][4]!=rr[k-1][4]+1 for k in range(1,len(rr)))
        if 24<=len(rr)<=160 and med>.95 and max(r[2] for r in rr)>30000 and not gaps and not any(r[7] for r in rr):high.append((abs(len(rr)-48),-med,s))
        gain=max(r[2] for r in rr)-rr[0][2]
        if 20<=len(rr)<=150 and rr[0][2]<4096 and gain>60000 and not gaps:growth.append((-gain,s))
        for r in pp:
            prev=rows[r[9]]
            if r[2]>10000 and prev[2]>30000 and r[4]==prev[4]+1 and r[5]<min(r[2],prev[2])*.25:drop.append((r[5]/min(r[2],prev[2]),r[8],s))
    high_sid=sorted(high)[0][2];growth_sid=next(s for _,s in sorted(growth) if s!=high_sid);drop_item=next(x for x in sorted(drop) if x[2] not in [high_sid,growth_sid])
    specs=[dict(key='reuse',title='工具循环：大部分输入延续上一轮',sid=high_sid,rule='无历史重置、保留调用连续，24–160 条请求且相邻前缀重合率中位数超过 95%；优先选择约 48 条请求的会话。'),dict(key='growth',title='上下文增长：输入越来越长',sid=growth_sid,rule='起始输入小于 4K、20–150 条连续调用，选择输入增长超过 60K 的会话；展示前 12 条保留调用。'),dict(key='change',title='前缀变化：相邻调用也可能不重合',sid=drop_item[2],rule='选择保留调用编号连续、前后输入都较长但共同前缀小于较短输入 25% 的片段；这只是变化案例，不代表普遍比例。')]
    selected=set()
    for spec in specs:
        ids=groups[spec['sid']]
        start=max(0,len(ids)//2-6) if spec['key']=='reuse' else 0
        if spec['key']=='change':start=max(0,ids.index(drop_item[1])-5)
        spec['ids']=ids[start:start+12]
        selected.update(spec['ids'])
    peak_specs=[]
    for key,bin_no in [('sessions',int(active.argmax())),('requests',int(counts.argmax()))]:
        ids=np.flatnonzero(bins==bin_no).tolist();session_order=list(dict.fromkeys(rows[i][0] for i in ids))
        peak_specs.append(dict(key=key,bin=bin_no,start_ms=bin_no*300000,end_ms=(bin_no+1)*300000,requests=len(ids),active_sessions=len(session_order),lanes=[dict(label='会话 '+str(j+1),session=si,source_id=sessions[si]['id'],calls=[dict(t=rows[i][1],input=rows[i][2],output=rows[i][3],call=rows[i][4],prefix=rows[i][5]) for i in ids if rows[i][0]==si]) for j,si in enumerate(session_order)]))
        if key=='sessions':
            cross_ids=[next(i for i in ids if rows[i][0]==si) for si in session_order[:8]];selected.update(cross_ids)
    selection=FOLDER/'selected_rows.json';selection.write_text(json.dumps(sorted(selected)))
    extracted=FOLDER/'selected_requests.json'
    if not extracted.exists() or {r['trace_row'] for r in json.loads(extracted.read_text())}!=selected:
        subprocess.run([str(TOOL),'extract',str(BUNDLE),str(selection),str(extracted)],check=True)
    lookup={r['trace_row']:r for r in json.loads(extracted.read_text())}
    from prepare_swe_chat import extract_calls
    snippets={};revisions={}
    for spec in specs:
        sm=sessions[spec['sid']];path=UPSTREAM/sm['path'];calls=extract_calls(path,sm['id'],Counter());by_response={r['response_id']:r for r in calls}
        for row_id in spec['ids']:
            source=lookup[row_id]['source'];call=by_response.get(source['response_id']);assert call is not None,'source response not recovered'
            delta=[]
            for event in call['input_delta']:
                c=event['content'];kind='工具结果' if isinstance(c,list) and any(b.get('type')=='tool_result' for b in c if isinstance(b,dict)) else '用户 / 环境消息'
                text=shorten(content_text(c));
                if text:delta.append(dict(kind=kind,text=text))
            tools=[b.get('name','unknown') for e in call['response_events'] for b in e['content'] if isinstance(e['content'],list) and isinstance(b,dict) and b.get('type')=='tool_use']
            snippets[row_id]=dict(delta=delta[:3],assistant=shorten('\n'.join(content_text(e['content']) for e in call['response_events']),200),tools=list(dict.fromkeys(tools)),history_reset=call['history_reset'])
        spec['initial_prompt']=shorten('\n'.join(content_text(e['content']) for e in calls[0]['input_delta']),360) if calls else ''
        meta=UPSTREAM/'.cache/huggingface/download'/(sm['path']+'.metadata')
        revisions[sm['path']]=meta.read_text().splitlines()[0] if meta.exists() else None
    def prefix(a,b):
        aa=lookup[a]['request'];bb=lookup[b]['request'];v=min(lcp(aa['hash_ids'],bb['hash_ids']),aa['input_length'],bb['input_length']);return v
    examples=[]
    for spec in specs:
        sm=sessions[spec['sid']];rr=[rows[i] for i in groups[spec['sid']]];valid=[r[5]/r[2] for r in rr if r[5] is not None]
        calls=[]
        for row_id in spec['ids']:
            r=rows[row_id];calls.append(dict(row=row_id,index=r[4],timestamp=r[1],input=r[2],output=r[3],prefix16=r[5],prefix64=r[6],gap_seconds=(r[1]-rows[r[9]][1])/1000 if r[9] is not None else None,previous_call=rows[r[9]][4] if r[9] is not None else None,reset=r[7],source_model=lookup[row_id]['source']['model'],**snippets[row_id]))
        matrix=[[prefix(a,b) if j<i else None for j,b in enumerate(spec['ids'])] for i,a in enumerate(spec['ids'])]
        examples.append(dict(key=spec['key'],title=spec['title'],rule=spec['rule'],session_id=sm['id'],transcript_path=sm['path'],total_calls=len(rr),median_overlap=float(np.median(valid)),span_ms=sm['last_ms']-sm['first_ms'],initial_prompt=spec['initial_prompt'],calls=calls,matrix=matrix))
    cross=[dict(label='会话 '+str(i+1),input=rows[row_id][2],call=rows[row_id][4],timestamp=rows[row_id][1],row=row_id) for i,row_id in enumerate(cross_ids)]
    cross_matrix=[[prefix(a,b) for b in cross_ids] for a in cross_ids]
    ratio_hist=np.histogram(ratio,bins=np.linspace(0,1,11))[0].astype(int).tolist()
    gaps=np.array([(r[1]-rows[r[9]][1])/1000 for r in pair])
    origin=sessions[rows[0][0]]['first_response_at'];rev=(UPSTREAM/'.cache/huggingface/download/README.md.metadata').read_text().splitlines()[0]
    provenance=dict(dataset='SALT-NLP/SWE-chat',dataset_url='https://huggingface.co/datasets/SALT-NLP/SWE-chat',revision=rev,license='ODC-By (upstream metadata: odc-by)',authors='SWE-chat authors: Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, Sanmi Koyejo (2026)',transcript_revisions=revisions,bundle=str(BUNDLE),manifest_sha256=hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest(),request_sha256=manifest['files']['requests.jsonl']['sha256'],source_sha256=manifest['files']['source_requests.jsonl']['sha256'],modifications='GLM-5.2 tokenization and 128K filtering, block-prefix comparisons and time aggregates; short transcript excerpts truncated for display; neither original model timing nor a real cache-hit measurement.')
    payload=dict(generated=datetime.now(timezone.utc).isoformat(),origin_utc=origin,provenance=provenance,scope=manifest['scope'],timestamp_policy=manifest['timestamp_policy'],history_policy=manifest['history_policy'],stats=dict(requests=len(rows),sessions=len(sessions),span_days=float(t[-1]/86400000),input_tokens=int(inputs.sum()),output_tokens=int(outputs.sum()),pairs=len(pair),weighted_prefix16=matched/paired_input,weighted_prefix64=matched64/paired_input,median_pair_overlap=float(np.median(ratio)),pairs_at_least_90=float(np.mean(ratio>=.9)),pairs_at_least_99=float(np.mean(ratio>=.99)),nonadjacent_pairs=raw['nonadjacent_call_pairs'],nonmonotone_pairs=raw['nonmonotone_call_pairs'],input_p50=float(np.median(inputs)),input_p95=float(np.percentile(inputs,95)),median_gap_seconds=float(np.median(gaps)),p95_gap_seconds=float(np.percentile(gaps,95))),daily=[[i,int(day_counts[i]),int(day_active[i]),int(day_input[i]),int(day_output[i])] for i in range(len(day_counts))],overlap_histogram=ratio_hist,examples=examples,peaks=peak_specs,cross=cross,cross_matrix=cross_matrix)
    assert sum(ratio_hist)==len(pair) and sum(x[1] for x in payload['daily'])==len(rows)
    (FOLDER/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    (FOLDER/'gallery_data.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    (FOLDER/'validation.json').write_text(json.dumps(dict(status='passed',requests=len(rows),sessions=len(sessions),selected_rows=len(selected),checks=['request/source rows aligned','manifest request and input-token totals agree','overlap histogram counts conserved','daily counts conserved','selected response IDs recovered from original logs','matrices use chain-aware prefix matches, not bag-of-block overlap']),indent=2))
    template=Path(__file__).with_name('swe_trace_gallery.html').read_text();assert template.count('__DATA__')==1
    page=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    (BASE/'swe_trace_examples.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(requests=len(rows),sessions=len(sessions),examples=[x['session_id'] for x in examples],weighted_prefix16=matched/paired_input),ensure_ascii=False),flush=True)
    print('Saved',BASE/'swe_trace_examples.html',flush=True)


if __name__=='__main__':main()
