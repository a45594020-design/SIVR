import argparse,csv,json
from collections import defaultdict
import numpy as np
from .common import ROOT,write

def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(baseline=False):
    records=[]
    for p in sorted((ROOT/'runs').glob('*/metrics.json')):
        m=json.loads(p.read_text());c=m['config'];a=m['auxiliary'];rec=json.loads((p.parent/'reconstruction_diagnostics.json').read_text());v=next(x for x in rec if x['split']=='eval')
        manifest=p.parent/'run_manifest.json';parameters=json.loads(manifest.read_text())['model_parameters'] if manifest.exists() else ''
        records.append({'run':p.parent.name,'group':c['group'],'seed':c['seed'],'mismatch':c.get('mismatch',0),'modalities':'+'.join(c.get('modalities',['brain','image','text'])),'semantic_shuffle':bool(c.get('semantic_shuffle',False)),'aux_top1':a['top1'],'aux_top5':a['top5'],'predicted_classes':a['predicted_classes'],'largest_class_share':a['largest_class_share'],'VT_top1':m['metrics'].get('VT',{}).get('top1',''),'VT_top5':m['metrics'].get('VT',{}).get('top5',''),'brain_mse':v['brain_pca_mse'],'brain_correlation':v['brain_pca_row_correlation'],'unique_variance':v['unique_variance'],'true_graph_energy':v['true_graph_energy'],'elapsed_seconds':m['elapsed_seconds'],'reused':m.get('reused',False),'gate':v['structure'].get('gate',''),'projection_retained_norm_ratio':v['structure'].get('projection_retained_norm_ratio','')})
        records[-1]['model_parameters']=parameters
    if baseline:
        records=[r for r in records if r['run'].startswith('E4_B0_seed')]
        table(ROOT/'baseline_evaluation_audit.csv',records)
        write(ROOT/'baseline_audit_complete.json',{'three_seeds':len(records)==3,'auxiliary_frozen_alpha':1.,'original_SVC_unchanged':True,'interpretation':'Evaluation audit only; auxiliary protocol cannot establish original classifier cause or validate superiority.'});return
    injection=[r for r in records if r['run'].startswith('E4_')]
    table(ROOT/'injection_comparison.csv',injection)
    table(ROOT/'mismatch_curves.csv',[r for r in records if r['group'] in ['Loss','Spatial'] and r['modalities']=='brain+image+text' and not r['semantic_shuffle']])
    changes=[]
    for group in ['Loss','Spatial']:
        for seed in [0,1,2]:
            reference=ROOT/'runs'/f'E4_{group}_m0_seed{seed}'/'eval_brain_reconstruction.npz'
            if not reference.exists():continue
            reference=np.load(reference)['pca']
            for level,prefix in [(1,'E4'),(.5,'E5')]:
                p=ROOT/'runs'/f'{prefix}_{group}_m{level}_seed{seed}'/'eval_brain_reconstruction.npz'
                if not p.exists():continue
                value=np.load(p)['pca']
                changes.append({'group':group,'seed':seed,'mismatch':level,'trained_model_output_change_rms':float(np.mean((value-reference)**2)**.5),'meaning':'paired retraining change; includes optimization stochasticity'})
    table(ROOT/'mismatch_output_changes.csv',changes)
    table(ROOT/'all_runs.csv',records)
    if len(injection)==27 and not (ROOT/'embedding_selection.json').exists():
        means={(g,mm):np.mean([r['aux_top1'] for r in injection if r['group']==g and r['mismatch']==mm]) for g in ['B0','Loss','Spatial','Hybrid'] for mm in ([0] if g=='B0' else [0,1])}
        candidates=[g for g in ['Loss','Spatial','Hybrid'] if means[(g,0)]>means[(g,1)] and means[(g,0)]>means[('B0',0)]]
        candidates.sort(key=lambda g:(-means[(g,0)],['Loss','Spatial','Hybrid'].index(g)))
        selected=candidates[0] if candidates else 'Spatial'
        write(ROOT/'embedding_selection.json',{'selected_for_modality_diagnostic':selected,'eligible':candidates,'fallback_diagnostic_only':not bool(candidates),'means':{f'{g}_{mm}':float(v) for (g,mm),v in means.items()},'rule_frozen_before_new_validation':True,'confirmatory_claim':False})
        (ROOT/'embedding_decision.md').write_text(f'# 接入开发阶段判断\n\n模态诊断使用 {selected}。合格候选：{candidates}。'+('没有方式通过入围规则；Spatial仅作为预登记机制诊断候选，不作为优胜方法。' if not candidates else '这是单折三种子的探索性筛选，尚未完成其他折或独立确认。')+'\n\n来源差、B0差与容量/普通正则对照分别报告。原VT接近机会水平时，不仅凭辅助检索宣称修复原分类任务或物理嵌入优越。',encoding='utf-8')
    if (ROOT/'modality_index.json').exists():
        index=json.loads((ROOT/'modality_index.json').read_text());lookup={r['run']:r for r in records}
        rows=[dict(lookup[i['run']],modality_condition=i['modalities']) for i in index]
        table(ROOT/'modality_effects.csv',rows)
        selected=json.loads((ROOT/'embedding_selection.json').read_text())['selected_for_modality_diagnostic'];paired=[]
        for mods in ['B','BV','BT','BVT']:
            for seed in [0,1,2]:
                a=next(r for r in rows if r['modality_condition']==mods and r['seed']==seed and r['group']==selected)
                b=next(r for r in rows if r['modality_condition']==mods and r['seed']==seed and r['group']=='B0')
                paired.append({'modalities':mods,'seed':seed,'candidate':selected,'aux_top1_difference_pp':100*(a['aux_top1']-b['aux_top1']),'variance_difference':a['unique_variance']-b['unique_variance']})
        table(ROOT/'modality_paired_effects.csv',paired)
        (ROOT/'E4_E5阶段报告.md').write_text('# E4/E5第一折开发报告\n\n9接入组×3种子、2方式×3失配×3种子、4模态×2组×3种子及2次语义打乱诊断完成。共享单元用索引追溯，不重复计作训练。\n\n## 协议边界\n\n原VT SVC与fit-only辅助Ridge/cosine检索分列；B-only辅助评价包含额外训练语义映射。所有主训练100轮，三种子。仅第一折，其他折未运行，本报告不构成方式优越性确认。\n\n## 结果文件\n\ninjection_comparison.csv、mismatch_curves.csv、modality_effects.csv、modality_paired_effects.csv、gradient_conflict.csv及embedding_decision.md。完整模型/预测、投影保留、结构门、真实图能量和源置换记录随运行保存。\n',encoding='utf-8')
    gradients=[]
    for p in (ROOT/'runs').glob('*/gradient_trajectory.jsonl'):
        for line in p.read_text().splitlines():
            g=json.loads(line)
            gradients.append({'run':p.parent.name,'epoch':g.get('epoch'),'task_gradient_norm':g.get('task_gradient_norm',''),'prior_gradient_norm':g.get('prior_gradient_norm',''),'weighted_prior_to_task_gradient':g.get('weighted_prior_to_task_gradient',''),'cosine':g.get('cosine',''),'gate':g.get('structure',{}).get('gate',''),'delta_rms':g.get('structure',{}).get('delta_rms','')})
    table(ROOT/'gradient_conflict.csv',gradients)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-only',action='store_true');a=p.parse_args();main(a.baseline_only)
