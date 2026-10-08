"""PC3-v18 runner: v17 rollout machinery with the sole edge-frontier repair."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
from typing import Any, Sequence
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
PROTOCOL=json.loads((HERE/'protocol.json').read_text(encoding='utf-8')); SCHEMA='pc3-v18-discriminating-selector-baselines/v1'
V17=HERE.parent/'pc3_v17_matched_selector_baselines'/'run_selector_baselines.py'
if str(V17.parent) not in sys.path: sys.path.insert(0,str(V17.parent))
spec=importlib.util.spec_from_file_location('pc3_v17_frozen_for_v18',V17)
if spec is None or spec.loader is None: raise RuntimeError(f'cannot load v17 runner: {V17}')
base=importlib.util.module_from_spec(spec); sys.modules[spec.name]=base; spec.loader.exec_module(base)
rules_spec=importlib.util.spec_from_file_location('pc3_v18_selector_rules',HERE/'selector_rules.py')
if rules_spec is None or rules_spec.loader is None: raise RuntimeError(f'cannot load v18 selector rules: {HERE / "selector_rules.py"}')
rules=importlib.util.module_from_spec(rules_spec); sys.modules[rules_spec.name]=rules; rules_spec.loader.exec_module(rules)
# Rebind only the v17 extension points; its rollout/update/evaluation body stays frozen.
base.PROTOCOL=PROTOCOL; base.SCHEMA=SCHEMA; base.v13.PROTOCOL=PROTOCOL; base.common_frontier=rules.common_frontier; base.choose_random=rules.choose_random; base.choose_uncertainty=rules.choose_uncertainty; base.choose_es=rules.choose_es; base.v13._select_target=base._select_target
run_acquisition,run_evaluation,run_merge=base.run_acquisition,base.run_evaluation,base.run_merge
def _configure_fixture(fixture: str) -> dict[str, Any]:
    if fixture not in PROTOCOL['fixtures']: raise ValueError(f'unknown fixture: {fixture}')
    config=dict(PROTOCOL); selected=dict(PROTOCOL['fixtures'][fixture])
    config.update(selected); config['fixture']=fixture
    base.PROTOCOL=config; base.v13.PROTOCOL=config
    base._initial_graph=lambda: base.v13.v11.initial_graph('U', selected['corruption'])
    base.v13._initial_eval_graph=lambda arm: base.v13.v11.initial_graph(arm, selected['corruption'])
    return selected
def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=('acquire','evaluate','merge')); parser.add_argument('--fixture',choices=('p1','p2'),default='p2'); parser.add_argument('--controller',type=int); parser.add_argument('--stream',type=int,choices=base.v13.STREAMS); parser.add_argument('--output-root',type=Path,required=True); parser.add_argument('--device',default='cuda'); parser.add_argument('--until-world',type=int); parser.add_argument('--world-start',type=int); parser.add_argument('--world-end',type=int); args=parser.parse_args(argv)
    fixture=_configure_fixture(args.fixture); output_root=args.output_root/args.fixture
    if args.stage in {'acquire','evaluate'} and (args.controller is None or args.stream is None): parser.error('controller and stream are required')
    if args.stage=='acquire': result=run_acquisition(controller=args.controller,stream=args.stream,output_root=output_root,device_name=args.device,until_world=args.until_world)
    elif args.stage=='evaluate': result=run_evaluation(controller=args.controller,stream=args.stream,output_root=output_root,device_name=args.device,world_start=args.world_start or fixture['evaluation_worlds'][0],world_end=args.world_end or fixture['evaluation_worlds'][1])
    else: result=run_merge(output_root=output_root)
    print(json.dumps(result,sort_keys=True,allow_nan=False)); return 0
if __name__=='__main__': raise SystemExit(main())
