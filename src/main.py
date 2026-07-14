import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

from report import generate_report

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_stage(script_name, *args):
    script_path = os.path.join(SRC_DIR, script_name)
    subprocess.run([sys.executable, script_path, *args], check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        collect_path = os.path.join(tmp_dir, "collected.json")
        forecast_path = os.path.join(tmp_dir, "forecast.json")

        # 1단계(데이터 수집·감성분석)와 2단계(Chronos 가격 예측)를 완전히 분리된
        # 프로세스로 실행한다. 감성분류기가 로드된 프로세스가 살아있는 채로 같은
        # 프로세스에서 이어 Chronos-Bolt-base까지 로드하면 이 환경에서 간헐적으로
        # 크래시가 발생해, 1단계 프로세스가 완전히 끝나 메모리를 반환한 뒤 2단계를
        # 새 프로세스로 띄운다 (각 스크립트 상단 설명 참고).
        print("=== 1단계: 데이터 수집 · 감성분석 (별도 프로세스) ===")
        _run_stage("stage_collect.py", collect_path)

        print("\n=== 2단계: 가격 예측 · 백테스트 (별도 프로세스) ===")
        _run_stage("stage_forecast.py", collect_path, forecast_path)

        with open(collect_path, encoding="utf-8") as f:
            collected = json.load(f)
        with open(forecast_path, encoding="utf-8") as f:
            forecasted = json.load(f)

    price_df = pd.DataFrame(collected["price_df"])
    related_prices = {name: pd.DataFrame(rows) for name, rows in collected["related_prices"].items()}

    print("\n=== 3단계: 리포트 생성 ===")
    report_path = generate_report(
        price_df,
        collected["news"],
        collected["sentiment_scores"],
        forecasted["forecast"],
        forecasted["backtest_result"],
        forecasted["rolling_result"],
        related_prices,
        collected["correlations"],
        collected["move_diagnosis"],
        collected["index_scan_result"],
    )
    print(f"  -> 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
