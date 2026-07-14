# hynix-stock-prediction

SK하이닉스(000660.KS) 주가를 사전학습된 시계열 예측 모델(Amazon Chronos)과 뉴스 감성분석으로 예측해보는 PoC입니다. 매일 실행하면 최신 데이터로 정적 대시보드(`docs/index.html`)를 다시 생성하고, GitHub Pages로 바로 배포됩니다.

**🔗 바로 보기: https://jihoonlee91.github.io/jihoonlee91-hynix-stock-prediction/**

> ⚠️ 교육/PoC 목적의 프로젝트이며 투자 판단에 사용하면 안 됩니다.

## 구성

| 모듈 | 역할 |
| --- | --- |
| `src/fetch_price.py` | `yfinance`로 SK하이닉스 일별 종가 수집 |
| `src/fetch_news.py` | Google News RSS로 "SK하이닉스"/"반도체 지정학" 뉴스 수집 (API 키 불필요) |
| `src/sentiment.py` | 사전학습 한국어 감성분류 모델로 뉴스 제목 감성 점수화 |
| `src/forecast.py` | Amazon Chronos(사전학습 시계열 파운데이션 모델)로 zero-shot 주가 예측 + 뉴스 감성 보정 |
| `src/report.py` | 예측 결과를 차트(PNG)와 정적 HTML 대시보드로 생성 |
| `src/main.py` | 위 파이프라인 전체를 순서대로 실행하는 진입점 |

왜 이런 모델들을 골랐는지, 어떤 제약 때문에 계획을 바꿨는지는 [`ARCHITECTURE.md`](ARCHITECTURE.md)에 정리했습니다.

## 실행 방법

```
pip install -r requirements.txt
python src/main.py
```

실행하면 `docs/index.html`과 `docs/assets/chart.png`가 새로 생성됩니다. `git commit` 후 push하면 GitHub Pages에 그대로 반영됩니다.

## GitHub Pages 배포 설정 (최초 1회)

1. 저장소 **Settings → Pages**로 이동
2. **Source**: `Deploy from a branch` 선택
3. **Branch**: `main` / `docs` 폴더 선택 후 Save

이후에는 `docs/` 폴더 내용을 커밋해서 push할 때마다 자동으로 페이지가 갱신됩니다.
