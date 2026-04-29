# Release Policy

이 문서는 정산 OCR 결과를 실제 원장에 반영하기 전 지켜야 할 운영 기준입니다.

## Manual Review Hard Gate

수동 검토 대상으로 지정된 참가자는 OCR 계좌 후보가 high confidence여도 자동 반영하지 않습니다.

해당 기준에 걸린 행은 자동 입력 대상이 아니라 review queue 대상입니다. 운영자는 원본 이미지에서 예금주, 계좌번호, 은행명, 대상자 일치 여부를 직접 확인한 뒤에만 반영할 수 있습니다.

Hard gate 대상:

- `REMAINING_REVIEW` 시트에 있는 참가자
- `auto_fill_openai_reranker`처럼 reranker가 자동 선택한 행 중 release gate 수동검수 큐에서 아직 confirmed가 아닌 행
- release gate의 `manual_autofill_review_queue.csv`에서 `review_status`가 `confirmed`가 아닌 행
- `reject_holder_mismatch`, `reject_not_bankbook`처럼 원본 문서 또는 예금주 불일치가 기록된 행

`no_source_file_found`는 새 원본 파일이 들어온 경우에만 별도 수동 확인 후 해제할 수 있습니다.

## Auto-Fill Criteria

자동 반영은 아래 조건을 모두 만족할 때만 허용합니다.

- OCR 후보가 계좌번호 정책 필터를 통과함
- reranker threshold와 margin을 통과함
- 대상자가 manual review hard gate에 걸리지 않음
- release gate에서 pending manual review가 0건임
- human eval report 기준 `wrong_positive=0`, `review_false_positive=0`, `safe_selection_precision=1.0`을 만족함
- release bundle PII scan이 통과함

## Open-Weight Reranker Contract

Open-weight structured reranker는 PII 원문 처리기가 아니라 redacted 후보 선택기입니다. OpenAI gpt-oss 같은 open-weight 모델은 로컬 Ollama/Transformers 실행 경로로 붙이고, 외부 API 호출을 전제로 하지 않습니다.

Reranker payload에 포함할 수 있는 항목:

- `source_id`
- `candidate_id`
- 계좌 형태 정보(`account_shape`, digit/group/hyphen count)
- 로컬 정책 점수와 boolean context/risk flag

Reranker payload에 포함하면 안 되는 항목:

- raw 계좌번호
- 사람 이름
- 원본 OCR 텍스트
- 원본 파일명 또는 로컬 경로
- human label workbook의 raw 정답

Reranker 응답은 structured JSON decision만 허용합니다. `accept` 결정은 `candidate_id`만 반환하고, raw 계좌번호 복원은 local-only `candidate_raw_map_local.jsonl`에서 수행합니다. `REMAINING_REVIEW` 대상은 reranker confidence가 높아도 `manual_review_hard_gate`로 강제 전환합니다.

## PII Handling

은행명은 검수 메타데이터이며 별도 PII 라벨링 대상은 아닙니다. 다만 이름, 마스킹 계좌, 은행명이 결합된 검수 CSV는 운영 데이터로 보고 접근권한을 제한합니다.

배포 번들에 포함하면 안 되는 항목:

- raw 계좌번호
- raw OCR 텍스트
- `candidate_features_local.jsonl`
- `candidate_raw_map_local.jsonl`
- 원본 통장사본/신분증 파일
- 로컬 파일 경로
- API token 또는 HF token

Release gate는 기본 PII 패턴 외에도 `--sensitive-workbook`에서 읽은 참가자 이름을 exact term으로 스캔합니다. 매칭 결과에는 파일 경로와 count만 남기고, 민감 용어 자체는 report에 기록하지 않습니다.

## Release Decision

제품 목표는 완전 자동 PII 추출/후보 선택까지 확장할 수 있지만, 원장 자동 반영은 별도 위험 등급으로 본다. PII 추출 성능 평가는 human label 검증셋으로 자동화하고, 실제 지급/원장 반영 정책은 hard gate와 release gate 결과를 따른다.

출시 가능 범위:

- 내부 운영자용 assisted release: 가능
- redacted open-weight reranker 기반 PII 추출 eval: 가능
- hard gate가 포함된 제한 배포: 가능
- hard gate 없이 고객 원장에 완전 자동 반영: 불가
