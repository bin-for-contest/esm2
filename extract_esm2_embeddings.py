import os
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. 설정 (경로 및 하이퍼파라미터)
CSV_PATH = "/mss5/knh0107/Bin/workspace/data/BRCA1_HUMAN_Findlay_2018.csv"
OUT_PARQUET = "/mss5/knh0107/Bin/workspace/brca1_esm2_features.parquet"
WINDOW_HALF = 510  # 변이 위치 기준 앞뒤 510 aa (총 최대 1021 aa -> ESM-2 1024 토큰 제한 준수)
BATCH_SIZE = 16    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 2. ProteinGym CSV 로드 및 변이 정보 파싱
df = pd.read_csv(CSV_PATH)
df["wt"] = df["mutant"].str[0]
df["pos"] = df["mutant"].str[1:-1].astype(int)  # 1-based position
df["mut"] = df["mutant"].str[-1]

# 야생형(WT) 전체 서열 복원 (첫 번째 행 기준)
first_row = df.iloc[0]
wt_seq_list = list(first_row["mutated_sequence"])
wt_seq_list[first_row["pos"] - 1] = first_row["wt"]
wt_full_seq = "".join(wt_seq_list)
seq_len = len(wt_full_seq)
print(f"[INFO] Loaded {len(df)} variants. Full BRCA1 sequence length: {seq_len} aa")

# 3. ESM-2 (650M, 33 layers) 모델 로드
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()

def get_window_seq(full_seq: str, pos_1based: int, half_win: int):
  """변이 위치(1-based)를 중심으로 윈도우 서열과 윈도우 내 상대 인덱스(0-based) 반환"""
  idx_0 = pos_1based - 1
  start = max(0, idx_0 - half_win)
  end = min(len(full_seq), idx_0 + half_win + 1)
  sub_seq = full_seq[start:end]
  rel_idx = idx_0 - start  # sub_seq 내에서의 0-based 위치
  return sub_seq, rel_idx

# 4. 위치(pos)별 야생형(WT) 기준 임베딩 및 로그 확률 캐싱 (중복 연산 방지)
unique_positions = sorted(df["pos"].unique())
wt_cache = {}

print(f"[INFO] Extracting WT embeddings & log-probs for {len(unique_positions)} unique positions...")
with torch.no_grad():
  for i in tqdm(range(0, len(unique_positions), BATCH_SIZE)):
    batch_pos = unique_positions[i : i + BATCH_SIZE]
    batch_data = []
    rel_indices = []
        
    for p in batch_pos:
      sub_seq, rel_idx = get_window_seq(wt_full_seq, p, WINDOW_HALF)
      batch_data.append((f"wt_{p}", sub_seq))
      rel_indices.append(rel_idx)
              
    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
        
    # 마지막 33번째 레이어의 hidden representation 및 logits 추출
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_reps = results["representations"][33]  # [B, L+2, 1280]
    logits = results["logits"]                   # [B, L+2, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)
    for b_idx, p in enumerate(batch_pos):
      # +1 이유: 맨 앞 토큰이 <cls> 토큰이기 때문
      tok_pos = rel_indices[b_idx] + 1
      wt_vec = token_reps[b_idx, tok_pos, :].cpu().numpy().astype(np.float32)
      wt_lp = log_probs[b_idx, tok_pos, :].cpu()
      wt_cache[p] = (wt_vec, wt_lp)

# 5. 변이 서열(Mutant) 임베딩 및 LLR 계산
mut_embeddings = []
wt_embeddings = []
llr_scores = []
cos_sims = []

print("[INFO] Extracting Mutant embeddings and computing LLR...")
with torch.no_grad():
  for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch_df = df.iloc[i : i + BATCH_SIZE]
    batch_data = []
    meta_info = []
    
    for _, row in batch_df.iterrows():
      sub_mut_seq, rel_idx = get_window_seq(row["mutated_sequence"], row["pos"], WINDOW_HALF)
      batch_data.append((row["mutant"], sub_mut_seq))
      meta_info.append((row["pos"], row["wt"], row["mut"], rel_idx))
            
    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
        
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_reps = results["representations"][33]
        
    for b_idx, (pos, wt_aa, mut_aa, rel_idx) in enumerate(meta_info):
      tok_pos = rel_idx + 1
      mut_vec = token_reps[b_idx, tok_pos, :].cpu().numpy().astype(np.float32)
      wt_vec, wt_lp = wt_cache[pos]
            
      # LLR = log P(Mut | WT_context) - log P(WT | WT_context)
      wt_tok_id = alphabet.get_idx(wt_aa)
      mut_tok_id = alphabet.get_idx(mut_aa)
      llr = float(wt_lp[mut_tok_id] - wt_lp[wt_tok_id])
      
      # 야생형 벡터와 변이형 벡터 간 코사인 유사도
      cos_sim = float(np.dot(wt_vec, mut_vec) / (np.linalg.norm(wt_vec) * np.linalg.norm(mut_vec) + 1e-8))
      mut_embeddings.append(mut_vec)
      wt_embeddings.append(wt_vec)
      llr_scores.append(llr)
      cos_sims.append(cos_sim)

# 6. 결과 DataFrame 조립 및 Parquet 저장
mut_emb_matrix = np.vstack(mut_embeddings)  # (N, 1280)
emb_cols = [f"esm_emb_{d}" for d in range(mut_emb_matrix.shape[1])]
emb_df = pd.DataFrame(mut_emb_matrix, columns=emb_cols)

out_df = pd.concat([
    df[["mutant", "wt", "pos", "mut", "DMS_score", "DMS_score_bin"]].reset_index(drop=True),
    pd.DataFrame({"esm_llr": llr_scores, "esm_cos_sim": cos_sims}, dtype=np.float32),
    emb_df], axis=1)

out_df.to_parquet(OUT_PARQUET, index=False)
print(f"[DONE] Saved feature table to {OUT_PARQUET} | Shape: {out_df.shape}")

