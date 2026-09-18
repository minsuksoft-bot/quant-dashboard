import json
from datetime import datetime, timedelta, timezone

# 기존 Signal 계산 코드
exec(open("gpt_app.py", encoding="utf-8").read())

# gpt_app.py에서 생성된 signal 결과를 아래 형식으로 저장
# 실제 변수명이 다르면 다음 단계에서 맞춰 수정

signals = []

for item in recent_signals:
    signals.append({
        "ticker": item["ticker"],
        "date": str(item["date"]),
        "score": item.get("score")
    })

result = {
    "generated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "signals": signals
}

with open("signal.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
