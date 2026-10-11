"""以滚动时间验证选择标签置信度、行业权重和双塔模型容量。"""

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from material_quality.inference.model import build_serving_tower_matrices, MaterialQualityPredictor
from material_quality.sample.refresh_training_data import atomic_json
from material_quality.training.evaluation import evaluate_classifier, evaluate_top_fraction
from material_quality.training.evaluate_refresh_model import purged_masks
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.train_serving_model import _fit_serving_towers, run_training
from plan_logger import log_info, set_process_name


PROFILES = [(8,0.001),(32,0.01),(64,0.01),(64,0.1)]
MODES = ['uniform','exposure','exposure_industry']
MODE_NAMES = {'uniform':'均匀权重','exposure':'曝光置信度','exposure_industry':'曝光置信度加行业均衡'}


def training_weights(exposures: np.ndarray, industries: np.ndarray, mode: str) -> np.ndarray:
    """同等下调低曝光正负样本，行业权重只根据当前训练窗口计算。"""
    exposures = np.asarray(exposures,dtype=np.float64)
    if exposures.ndim!=1 or exposures.size==0 or not np.isfinite(exposures).all() or np.any(exposures<0):
        raise ValueError('曝光量必须为非负有限数且不能为空')
    if industries.shape!=exposures.shape or mode not in MODES:
        raise ValueError('行业长度或权重模式无效')
    weights = np.ones(exposures.size,dtype=np.float64)
    if mode!='uniform':
        weights = np.sqrt(np.clip(exposures/500.0,0.01,1.0))
    if mode=='exposure_industry':
        counts = Counter(industries.tolist())
        typical = float(np.median(list(counts.values())))
        weights *= np.array([np.clip(np.sqrt(typical/counts[k]),0.5,2.0) for k in industries])
    return weights/weights.mean()


def validation_folds(dates: np.ndarray, test_start: datetime.date) -> list:
    """调参验证标签也须在测试开始前闭合，保留额外七日间隔。"""
    folds = []
    for offset in [28,21,14]:
        start = test_start-datetime.timedelta(days=offset)
        end = start+datetime.timedelta(days=7)
        train,validation = purged_masks(dates,start,end)
        if not train.any() or not validation.any():
            raise ValueError('滚动时间窗口没有足够样本')
        if np.any(dates[validation]+7>test_start.toordinal()):
            raise ValueError('验证标签尚未在测试开始前闭合')
        folds.append({'start':start.isoformat(),'train':train,'validation':validation})
    return folds


def fit_probability(dataset: dict, exposures: np.ndarray, train: np.ndarray,
                    predict: np.ndarray, candidate: dict) -> np.ndarray:
    """投放结果仅决定训练损失权重，不进入模型输入特征。"""
    profile = {'pca_components':candidate['pca_components'],'c':candidate['c']}
    weights = training_weights(exposures[train],dataset['primary_industries'][train],candidate['mode'])
    models,_ = _fit_serving_towers({k:v[train] for k,v in dataset.items()},
                                   (dataset['labels'][train]>0).astype(np.int32),profile,weights)
    towers = build_serving_tower_matrices(dataset['features'][predict],dataset['script_features'][predict],
                                         dataset['script_features'].shape[1])
    return sum(0.5*models[k].predict_proba(towers[k])[:,1] for k in ['visual','script'])


def account_groups(dataset: dict, candidates: dict, plans: dict) -> np.ndarray:
    """只将账户唯一且全部计划有映射的素材纳入同账户排序指标。"""
    groups = []
    for material_id in dataset['material_ids']:
        item = candidates.get(str(material_id))
        ids = item['plan_ids'] if item else []
        values = {plans[str(p)] for p in ids if str(p) in plans}
        groups.append(next(iter(values)) if ids and len(values)==1 and all(str(p) in plans for p in ids) else '')
    return np.asarray(groups)


def within_account_metrics(labels: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict:
    """报告同账户正负样本排序，明确可评估账户及样本覆盖。"""
    aucs, pairs, samples = [], [], 0
    for group in sorted(set(groups)-{''}):
        mask = groups==group
        positive = int(labels[mask].sum());negative = int(mask.sum())-positive
        if positive<2 or negative<2:
            continue
        aucs.append(float(roc_auc_score(labels[mask],probability[mask])))
        pairs.append(positive*negative);samples += int(mask.sum())
    return {'eligible_accounts':len(aucs),'covered_samples':samples,
            'macro_auc':float(np.mean(aucs)) if aucs else None,
            'pair_weighted_auc':float(np.average(aucs,weights=pairs)) if aucs else None}


def paired_account_interval(labels: np.ndarray, new: np.ndarray, old: np.ndarray,
                            groups: np.ndarray, iterations: int = 1000) -> dict:
    """对账户唯一的素材按账户重抽样，避免同账户相关性低估不确定性。"""
    unique = sorted(set(groups)-{''})
    indices = [np.flatnonzero(groups==g) for g in unique]
    if len(unique)<2:
        return {'account_count':len(unique),'interval':None}
    rng = np.random.RandomState(20261011);deltas=[]
    for _ in range(iterations):
        selected = np.concatenate([indices[i] for i in rng.randint(0,len(unique),len(unique))])
        if np.unique(labels[selected]).size<2:
            continue
        deltas.append(roc_auc_score(labels[selected],new[selected])-roc_auc_score(labels[selected],old[selected]))
    return {'account_count':len(unique),'covered_samples':int((groups!='').sum()),
            'interval':np.quantile(deltas,[0.025,0.975]).tolist() if deltas else None}


def run(args: argparse.Namespace) -> dict:
    """先固定候选和选择规则，验证完成后才读取历史测试结果。"""
    output = Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    test_start = datetime.date.fromisoformat(args.test_from)
    protocol = {'test_from':args.test_from,'test_status':'已查看过的时间段，仅作改进实验，待新的独立时间段验收',
                'profiles':PROFILES,'weight_modes':MODE_NAMES,'confidence':'sqrt(clip(send/500,0.01,1))，正负同权规则',
                'industry_balance':'训练窗口内行业样本数平方根倒数，相对中位数截断到0.5至2倍',
                'selection':'三个滚动窗口非加权AUC均值减0.25倍标准差，仅按验证选择',
                'validation_offsets_days':[28,21,14],'label_window_days':7,
                'model_inputs':'只有画面、媒体和脚本特征；曝光、行业、账户不作为预测输入'}
    atomic_json(output/'protocol.json',protocol)
    dataset,manifest = load_script_augmented_dataset(args.snapshot_dir,args.script_feature_root,'codex_agent_script_cnclip_v1')
    exposure_source = json.loads(Path(args.exposure_path).read_text())
    if exposure_source['dataset_sha256']!=manifest['base_dataset_sha256']:
        raise ValueError('曝光记录与冻结快照摘要不一致')
    exposures = np.asarray([exposure_source['records'][str(mid)]['send_7d'] for mid in dataset['material_ids']],dtype=np.float64)
    labels = (dataset['labels']>0).astype(np.int32)
    folds = validation_folds(dataset['date_ordinals'],test_start)
    candidates=[]
    for mode in MODES:
        for dim,c in PROFILES:
            candidate={'name':'{}_pca{}_c{}'.format(mode,dim,c),'mode':mode,'pca_components':dim,'c':c}
            scores=[]
            for fold in folds:
                p=fit_probability(dataset,exposures,fold['train'],fold['validation'],candidate)
                scores.append(float(roc_auc_score(labels[fold['validation']],p)))
            candidate.update({'fold_aucs':scores,'validation_mean':float(np.mean(scores)),
                              'selection_score':float(np.mean(scores)-0.25*np.std(scores))})
            candidates.append(candidate)
            atomic_json(output/'validation_progress.json',{'completed':len(candidates),'total':len(MODES)*len(PROFILES),'candidates':candidates})
            log_info('候选验证完成：{}，平均AUC {:.4f}'.format(candidate['name'],candidate['validation_mean']))
    selected=max(candidates,key=lambda c:c['selection_score'])
    mode_winners=[max([c for c in candidates if c['mode']==mode],key=lambda c:c['selection_score']) for mode in MODES]
    atomic_json(output/'selection.json',{'selected':selected,'mode_winners':mode_winners,'candidates':candidates})
    train,test=purged_masks(dataset['date_ordinals'],test_start)
    source={x['material_id']:x for x in map(json.loads,Path(args.candidates_path).read_text().splitlines())}
    plans=json.loads(Path(args.plan_account_path).read_text());groups=account_groups(dataset,source,plans)[test]
    baseline=next(c for c in candidates if c['mode']=='uniform' and c['pca_components']==8)
    requested={c['name']:c for c in [baseline]+mode_winners};predictions={};results=[]
    for name,candidate in requested.items():
        p=fit_probability(dataset,exposures,train,test,candidate);predictions[name]=p
        adequate=exposures[test]>=500
        results.append({'name':name,'mode':candidate['mode'],'selected':name==selected['name'],
                        'test':evaluate_classifier(labels[test],p),
                        'top20':evaluate_top_fraction(dataset['labels'][test],p,.2),
                        'within_account':within_account_metrics(labels[test],p,groups),
                        'adequate_exposure_count':int(adequate.sum()),
                        'adequate_exposure':evaluate_classifier(labels[test][adequate],p[adequate])})
    comparison=paired_account_interval(labels[test],predictions[selected['name']],predictions[baseline['name']],groups)
    metrics={'protocol':protocol,'base_dataset_sha256':manifest['base_dataset_sha256'],
             'script_dataset_sha256':manifest['script_dataset_sha256'],'train_count':int(train.sum()),
             'test_count':int(test.sum()),'selected':selected,'results':results,'account_cluster_auc_delta':comparison}
    atomic_json(output/'metrics.json',metrics)
    np.savez_compressed(output/'test_predictions.npz',content_sha256=dataset['content_sha256'][test],
                        labels=labels[test],**predictions)
    weights=training_weights(exposures,dataset['primary_industries'],selected['mode'])
    weight_path=output/'full_training_weights.npz'
    np.savez_compressed(weight_path,content_sha256=dataset['content_sha256'],sample_weight=weights)
    model_dir=output/'candidate_model'
    run_training(argparse.Namespace(base_snapshot_dir=args.snapshot_dir,script_feature_root=args.script_feature_root,
        script_feature_version='codex_agent_script_cnclip_v1',output_dir=str(model_dir),model_version=args.model_version,
        pca_components=selected['pca_components'],regularization_c=selected['c'],sample_weight_path=str(weight_path),
        sample_weight_description=MODE_NAMES[selected['mode']]))
    model_manifest=json.loads((model_dir/'manifest.json').read_text());model_manifest.update({
        'release_status':'awaiting_fresh_holdout','evaluation_path':str(output/'metrics.json'),
        'selection_method':'按标签闭合的滚动时间验证选择；已有测试段仅作诊断'})
    atomic_json(model_dir/'manifest.json',model_manifest)
    card=(model_dir/'MODEL_CARD.md').read_text().replace('# 素材质量正式评分模型','# 素材质量候选模型（待独立验收）',1)
    card+='\n\n采用{}，PCA{}，C={}。本轮测试段此前已被查看，不能视为独立验收；未部署，尚未在目标Python 3.8环境验证。\n'.format(MODE_NAMES[selected['mode']],selected['pca_components'],selected['c'])
    (model_dir/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    ok,error=MaterialQualityPredictor.load(str(model_dir)).verify_smoke_test()
    if not ok:raise RuntimeError('候选模型回放失败')
    lines=['# 置信度加权模型改进实验','','已查看的历史测试段，仅用于比较改进，待新的独立时间段验收。','',
           '选择规则：训练和验证标签均在测试开始前闭合，按三个验证窗口平均AUC减波动惩罚选择。','',
           '| 方案 | 历史测试AUC | PR-AUC | 前20%正样本率 | 同账户宏平均AUC |','|---|---:|---:|---:|---:|']
    for item in results:
        lines.append('| {}{} | {:.4f} | {:.4f} | {:.2%} | {:.4f} |'.format(
            item['name'],'（验证选中）' if item['selected'] else '',item['test']['roc_auc'],item['test']['pr_auc'],
            item['top20']['positive_rate'],item['within_account']['macro_auc'] or 0))
    lines+=['','候选制品仅改变训练方式，线上输入仍为内容特征。曝光置信度为启发式权重，不代表已识别素材的因果质量。',
            '全部{}条样本均有经冻结标签核对的曝光记录；同账户指标仅使用账户唯一且每类至少两条的账户。'.format(len(labels)),
            '所选方案相对原配置的账户聚类AUC差值95%区间：{}。'.format(comparison['interval'])]
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    log_info('改进实验完成：{}，候选回放误差{}'.format(selected['name'],error))
    return metrics


def main() -> None:
    """读取冻结数据与曝光记录，运行本地改进实验。"""
    set_process_name('material_quality_confidence')
    parser=argparse.ArgumentParser(description='训练曝光置信度加权双塔分类器')
    for key in ['snapshot-dir','script-feature-root','exposure-path','candidates-path','plan-account-path','output-dir','model-version']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--test-from',default='2026-09-15')
    run(parser.parse_args())


if __name__=='__main__':
    main()
