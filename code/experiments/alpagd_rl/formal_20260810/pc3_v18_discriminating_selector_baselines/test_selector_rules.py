import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from selector_rules import common_frontier,choose_random,choose_uncertainty
class G:
 def probability(self,s,t): return .8 if (s,t) in {(1,0),(2,0)} else .1
class P:
 graph=G()
 def distribution(self,a,t): return [.3,.3,.3]
def test_edge_frontier_has_multiple_candidates_and_audit_fields():
 c=common_frontier(provider=P(),targets=('A','B','C'),achieved={'A':False,'B':False,'C':False},history={},world=1,used_masks=set(),used_pairs=set(),max_challenges=3)
 assert len(c)==2 and c[0]['frontier_size']==2 and c[0]['candidate_pair_keys']==['A|B','A|C']
def test_random_and_uncertainty_log_selected_score_and_probability():
 c=common_frontier(provider=P(),targets=('A','B','C'),achieved={'A':False,'B':False,'C':False},history={},world=1,used_masks=set(),used_pairs=set(),max_challenges=3)
 for value in (choose_random(c,.9),choose_uncertainty(c,.1)):
  assert 'selected_selector_score' in value and value['selected_evidence_edge_probability']==.1
def test_runner_explicitly_binds_its_own_rules_module():
 source=(HERE/'run_selector_baselines.py').read_text(encoding='utf-8')
 assert "HERE/'selector_rules.py'" in source
 assert 'base.common_frontier=rules.common_frontier' in source
def test_fixture_contract_preserves_p2_and_reserves_disjoint_p1_worlds():
 import json
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 assert protocol['fixtures']['p2']['acquisition_worlds']==[1440001,1440040]
 assert protocol['fixtures']['p2']['evaluation_worlds']==[1450001,1450100]
 assert protocol['fixtures']['p1']['corruption']=='P1'
 assert protocol['fixtures']['p1']['acquisition_worlds']==[1460001,1460040]
 assert "parser.add_argument('--fixture',choices=('p1','p2'),default='p2')" in (HERE/'run_selector_baselines.py').read_text(encoding='utf-8')
