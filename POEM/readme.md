# 변경점

POEM을 위해 Domain 판별기 수정.

- Domain Classifier도 Label Classifier와 똑같이 CNN 사용, out_features의 개수는 domain 수와 같다.
  Similarity Loss, Discrimination Loss는 이전과 같다.
- Similarity Loss는 Domain Feature와 Label Feature 두 개의 cosine similarity,
- Discrimination Loss는 Feature를 input으로 받고, domain의 feature일지 label의 feature일지 확률을 출력.
- 이전과 다름이 없다.

# 문제점

- 성능이 같음
- Label Classifier가 충분히 학습되고 수렴하는지 확인이 필요.
- 아니면, 모델 구조의 문제일지도...?
