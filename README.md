# sk-hynix-stock-prediction

SK하이닉스(000660.KS) 주가를 여러 사전학습·통계·머신러닝 모델과 국내외 뉴스 감성분석으로
함께 분석하는 정량 리포트입니다. 매일 실행하면 최신 데이터로 정적 대시보드(`docs/index.html`)를
다시 생성하고, GitHub Pages로 바로 배포됩니다.

**🔗 바로 보기: https://jihoonlee91.github.io/sk-hynix-stock-prediction/**

> ⚠️ 이 리포트는 공개 데이터와 공개 모델을 결합한 정량 분석 참고 자료이며, 투자 손익에 대한
> 법적 책임을 지지 않습니다. 최종 투자 판단과 책임은 이용자 본인에게 있습니다.

## 무엇을 보여주나

- **가격 예측**: Amazon Chronos-Bolt(시계열 파운데이션 모델) + 감쇠추세 지수평활을 앙상블하고,
  기술적 지표 기반 gradient boosting 모델을 비교축으로 함께 보여줍니다.
- **뉴스 감성**: 한국어 7개 + 영어 7개 검색어로 국내외(미국/일본/중국계 영문매체 포함) 뉴스를
  모아 중복 기사를 합치고, 언어별 금융 특화 모델로 감성을 채점해 예측에 반영합니다.
- **시장 맥락**: 코스피/S&P500/나스닥(시장 전체), 필라델피아반도체지수(섹터), 삼성전자/Micron/SanDisk
  (동종업계) 세 계층과 비교해 하이닉스 움직임의 원인이 시장·섹터·개별 이슈 중 무엇인지 진단하고,
  코스피200·나스닥100·S&P500 대표종목 중 상관관계가 높은 종목도 함께 보여줍니다.
- **성능 검증**: naive(직전값 유지) 기준·Chronos 단독·앙상블을 walk-forward 롤링 백테스트로
  비교하고, 방향 적중률과 실제-예측 수익률 산점도로 실전 유용성을 가늠할 수 있게 합니다.

## 구성

| 모듈 | 역할 |
| --- | --- |
| `src/fetch_price.py` | `yfinance`로 SK하이닉스 일별 종가 수집 |
| `src/related_stocks.py` | 코스피/S&P500/나스닥/반도체지수/동종업계 비교, 움직임 원인 진단 |
| `src/index_scan.py` | 코스피200/나스닥100/S&P500 대표종목 중 상관관계 상위 종목 스캔 |
| `src/fetch_news.py` | 국내외(한국어/영어) Google News RSS 수집 + 중복 기사 통합 |
| `src/sentiment.py` | 언어별 금융 특화 감성분류(가중평균: 확신도·검색어중요도·최신성) |
| `src/technical_indicators.py` | RSI/MACD/이동평균/볼린저밴드 등 표준 기술적 지표 계산 |
| `src/technical_model.py` | 기술적 지표 기반 gradient boosting 분위수 회귀 모델 |
| `src/forecast.py` | Chronos-Bolt + 추세모델 앙상블, 뉴스 감성 보정(시점 감쇠 적용) |
| `src/backtest.py` | naive/Chronos단독/앙상블 walk-forward 백테스트, 방향 적중률 |
| `src/report.py` | 차트(PNG)와 정적 HTML 리포트 생성 |
| `src/stage_collect.py`, `src/stage_forecast.py` | 무거운 모델 로딩 단계를 분리된 프로세스로 실행 |
| `src/main.py` | 전체 파이프라인 진입점 |

왜 이런 모델·방법론을 골랐는지는 [`ARCHITECTURE.md`](ARCHITECTURE.md)에 정리했습니다.

## 실행 방법

```
pip install -r requirements.txt
python src/main.py
```

실행하면 `docs/index.html`과 `docs/assets/*.png`가 새로 생성됩니다. `git commit` 후 push하면
GitHub Pages에 그대로 반영됩니다. 내부적으로 데이터 수집·감성분석(1단계)과 가격 예측(2단계)이
서로 다른 프로세스로 실행되는데, 두 단계가 로드하는 모델(BERT 계열 감성분류기 2종 + Chronos)을
같은 프로세스에서 이어서 로드하면 일부 환경에서 메모리 문제가 생겨 분리했습니다
(`ARCHITECTURE.md`의 "프로세스 분리" 항목 참고).

## GitHub Pages 배포 설정 (최초 1회)

1. 저장소 **Settings → Pages**로 이동
2. **Source**: `Deploy from a branch` 선택
3. **Branch**: `main` / `docs` 폴더 선택 후 Save

이후에는 `docs/` 폴더 내용을 커밋해서 push할 때마다 자동으로 페이지가 갱신됩니다.
