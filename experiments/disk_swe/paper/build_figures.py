"""Reproduce the paper figures from the frozen 32x / 16K result CSV."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'assets'
DATA=ROOT/'data'
COLORS=['#D55E00','#E69F00','#009E73','#0072B2','#7C5FA3']
DRAMS=[64,128,256,384,512,1000]
BANDWIDTHS=[1,2,4,8,16]


def main():
    rows=[]
    with (DATA/'minimax_32x_16k_results.csv').open() as f:
        for row in csv.DictReader(f):rows.append({k:float(v) for k,v in row.items()})
    assert len(rows)==30
    values={(int(r['dram_gb']),int(r['read_gbps'])):r for r in rows}
    assert len(values)==30 and all(r['request_count']==607190 for r in rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300})
    ASSETS.mkdir(exist_ok=True)
    def style(ax):
        ax.grid(axis='y',color='#dce2e3',linewidth=.6,zorder=0)
        ax.set_axisbelow(True)
    def capacity(ax):
        ax.set_xticks(DRAMS,[str(1024 if v==1000 else v) for v in DRAMS],rotation=35,ha='right')
        ax.set_xlim(35,1030);ax.set_xlabel('P-side DRAM capacity (GB)');style(ax)
    def save(fig,name):
        for extension in ['png','pdf','svg']:fig.savefig(ASSETS/f'{name}.{extension}',bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    # Fixed 8 GB/s avoids the lower reuse seen when slow restoration alters admission.
    fixed=[values[(d,8)] for d in DRAMS]
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
    axes[0].plot(DRAMS,[r['dram_reuse_pct'] for r in fixed],'-o',color='#009E73',label='DRAM direct hits',lw=1.8,ms=4)
    axes[0].plot(DRAMS,[r['ssd_reuse_pct'] for r in fixed],'-s',color='#D55E00',label='SSD-restored hits',lw=1.8,ms=4)
    axes[0].set_title('(a) First-admission cache hits');axes[0].set_ylabel('Input-token hit ratio (%)');axes[0].set_ylim(-.5,17.5);axes[0].legend(loc='upper center',frameon=False)
    axes[1].plot(DRAMS,[r['ssd_read_tb'] for r in fixed],'-D',color='#0072B2',lw=1.8,ms=4)
    axes[1].set_title('(b) Cumulative SSD reads');axes[1].set_ylabel('SSD read traffic\n(TB per TP shard)');axes[1].set_ylim(0,150)
    for ax in axes:capacity(ax)
    save(fig,'fig1_dram_ssd_reuse')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.0),layout='constrained')
    handles=[]
    for b,c,marker in zip(BANDWIDTHS,COLORS,['o','s','^','D','v']):
        h=axes[0].plot(DRAMS,[values[(d,b)]['adjusted_mean_s'] for d in DRAMS],color=c,marker=marker,label=f'{b} GB/s',lw=1.6,ms=3.6)[0];handles.append(h)
        axes[1].plot(DRAMS,[values[(d,b)]['adjusted_p99_s'] for d in DRAMS],color=c,marker=marker,lw=1.6,ms=3.6)
    axes[0].set_title('(a) Mean adjusted TTFT');axes[0].set_ylabel('Adjusted TTFT (s)');axes[0].set_ylim(0,.48)
    axes[1].set_title('(b) P99 adjusted TTFT');axes[1].set_ylabel('Adjusted TTFT (s)');axes[1].set_ylim(0,6.2)
    for ax in axes:capacity(ax)
    fig.legend(handles,[f'{b} GB/s' for b in BANDWIDTHS],ncol=5,loc='upper center',bbox_to_anchor=(.53,1.16),frameon=False,title='SSD read bandwidth per TP shard',title_fontsize=9)
    save(fig,'fig2_bandwidth_sensitivity')
    result=dict(status='passed',cases=30,selected_factor=32,budget_tokens=16384,speedup=1,bandwidths=BANDWIDTHS,dram_gb=DRAMS,
                checks=['complete 6 x 5 matrix','all cases completed 607190 requests','fixed-bandwidth reuse plot uses 8 GB/s only'],
                comparisons=dict(ssd_read_128_vs_512_at_8=values[(128,8)]['ssd_read_tb']/values[(512,8)]['ssd_read_tb'],
                                 p99_penalty_64_vs_512_at_1=values[(64,1)]['adjusted_p99_s']-values[(512,1)]['adjusted_p99_s'],
                                 p99_penalty_64_vs_512_at_8=values[(64,8)]['adjusted_p99_s']-values[(512,8)]['adjusted_p99_s']))
    (DATA/'figure_validation.json').write_text(json.dumps(result,indent=2))
    print('Generated two figures in PNG/PDF/SVG; selected data validated.')


if __name__=='__main__':main()
