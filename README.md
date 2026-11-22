# esm2

mutant	변이 식별자	ProteinGym의 고유 변이 표기(예: C39Y = 39번 시스테인(C)이 타이로신(Y)으로 치환)
wt	야생형 잔기	변이 전 원래(Wild-Type) 아미노산 한 글자 코드(예: C)

pos	변이 위치	BRCA1 단백질 전체 아미노산 서열(1~1,863) 중 변이가 일어난 1-based 잔기 번호입니다. 교차검증(GroupKFold) 시 데이터 누수를 막는 그룹 기준

mut	변이형 잔기	치환된 후(Mutant)의 아미노산 한 글자 코드(예: Y)

DMS_score	regression target	Findlay et al. (2018) 실험에서 측정된 연속적인 기능 점수. 보통 0에 가까우면 정상(기능 유지), 음수로 크게 떨어질수록(예: 2.0) 기능 상실(손상)을 의미

DMS_score_bin	classification 정답	DMS_score를 특정 임계값 기준으로 나눈 이진 라벨. (1: 기능 유지/정상, 0: 기능 손상/병원성). 모델이 예측한 기능 점수에 컷오프를 적용해 평가할 때 정답지

esm_llr	서열 feature(scalar)	Log-Likelihood Ratio (logP(Mut)−logP(WT))
ESM-2 언어 모델이 보기에 원래 아미노산 대비 바뀐 아미노산이 진화적으로 얼마나 자연스러운지를 나타냅니다. 음수가 클수록 해로운 변이일 가능성이 높음

esm_cos_sim	서열 feature(scalar)	변이 전(WT) 히든 벡터와 변이 후(Mut) 히든 벡터 사이의 코사인 유사도 1.0에 가까울수록 아미노산이 바뀌어도 주변 서열 문맥 표현이 거의 변하지 않았음

esm_emb_0	서열 feature(vector)	변이 후 서열(mutated_sequence)을 ESM-2(650M)의 마지막 33번째 레이어에 통과시켰을 때, 해당 pos 위치에서 나온 1,280차원 히든 벡터의 0번째 원소

esm_emb_1	서열 feature(vector)	위 1,280차원 히든 벡터의 1번째 원소입니다. (뒤이어 esm_emb_2부터 esm_emb_1279까지 총 1,280개 열이 이어져 있습니다.)
