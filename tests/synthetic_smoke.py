"""Execute the shipped pure numerical function without optional model libraries."""
import ast
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
source=(root/'original_rsp_raw_utils.py').read_text()
ns={'np':np}
for name in ['margin_rows','rsp_components']:
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile('from __future__ import annotations\n'+ast.get_source_segment(source,node),str(root/'original_rsp_raw_utils.py'),'exec'),ns)
clean=np.array([[.7,.3],[.7,.3]])
perturbed=np.array([[[.4,.6],[.6,.4]],[[.2,.8],[.55,.45]]])
result=ns['rsp_components'](clean,perturbed,.5)
np.testing.assert_allclose(result['flip'],[1,0])
np.testing.assert_allclose(result['drop'],[0,.25])
np.testing.assert_allclose(result['rsp'],[1,.125])
print('PASS: shipped RSP flip/nonflip contraction function, synthetic probabilities only')
import json
source=(root/'train_original_rsp_raw.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='select_exact')
exec(compile('from __future__ import annotations\n'+ast.get_source_segment(source,node),str(root/'train_original_rsp_raw.py'),'exec'),ns)
pool=[{'uid':f'synthetic-{i}'} for i in range(3)]
scores=[{'uid':x['uid'],'rsp_synthetic':v} for x,v in zip(pool,[1.,1.,-1e30])]
assert ns['select_exact'](pool,scores,'rsp_synthetic',2)==[0,1]
try:ns['select_exact'](pool,scores,'rsp_synthetic',3)
except RuntimeError:pass
else:raise AssertionError('Selection escaped entropy prefilter')
config=json.loads((root/'docs/paper_config.json').read_text())
module=ast.parse((root/'make_original_rsp_expanded_dev_lambda_manifest.py').read_text())
grid=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GLOBAL_GRID' for t in n.targets))
assert list(grid)==config['lambda_grid']
print('PASS: stable tie-break, prefilter guard and final paper lambda grid')
import importlib.util
p=root/'analysis/recompute_submission_sensitivities.py'
spec=importlib.util.spec_from_file_location('synthetic_final_analysis',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.normalized_question('  SYNTHETIC   Question\n')=='synthetic question'
probs=np.array([[.9,.1,0],[.2,.8,0]])
accuracy,f1=module.metrics(probs,np.array([0,1]))
assert accuracy==1.0 and np.isclose(f1,2/3)
print('PASS: final-analysis question normalization and fixed-label-space Macro-F1')
