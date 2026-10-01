"""Freeze the preliminary TUSZ external subset result into release metadata."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'outputs/reports/v3-compact-extension-v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=R/'tusz_eval_subset_v1_results_manifest.json'; result=json.loads(rp.read_text())
 if result['status']!='preliminary_frozen_external_subset_complete': raise RuntimeError('result not complete')
 sp=R/'tuh_external_status.json'; status=json.loads(sp.read_text()); status.update({'updated_utc':datetime.now(timezone.utc).isoformat(),'status':'preliminary_frozen_external_subset_complete','external_subset_validation':{'cohort_id':'tusz-eval-subset-v1','patients':10,'records':198,'duration_h':26.955833333333334,'manifest':'tusz_eval_subset_v1_manifest.json','manifest_sha256':result['cohort_manifest_sha256'],'results_manifest':rp.name,'results_manifest_sha256':sha(rp),'status':result['status'],'full_tusz_eval':False},'external_performance_reported':True,'model_retraining':False,'threshold_retuning':False,'next_gate':'audit patient-level waterfall and calibration, then decide whether to expand to all 43 TUSZ eval patients','notes':'Preliminary frozen external subset results are complete; current CHB test remains locked and no TUSZ tuning was performed.'}); sp.write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
 lp=R/'release_manifest.json'; release=json.loads(lp.read_text()); release['status']='h4_preliminary_negative_h5_descriptive_complete_tuh_subset_external_complete'; tuh=release.setdefault('analyses',{}).setdefault('tuh_external_validation',{}); tuh.update({'status':status['status'],'status_file':f'outputs/reports/v3-compact-extension-v1/{sp.name}','status_sha256':sha(sp),'external_subset_results':f'outputs/reports/v3-compact-extension-v1/{rp.name}','external_subset_results_sha256':sha(rp),'external_performance_reported':True,'full_tusz_eval':False,'retrained':False,'threshold_retuned':False}); release['remaining_work']=[x for x in release.get('remaining_work',[]) if x!='tuh-frozen-external-validation']+['tuh-full-eval-optional-confirmation']; lp.write_text(json.dumps(release,indent=2,sort_keys=True)+'\n'); print(json.dumps({'release_status':release['status'],'tuh_status':status['status'],'results':result['results'],'next_gate':status['next_gate']},indent=2))
if __name__=='__main__': main()
