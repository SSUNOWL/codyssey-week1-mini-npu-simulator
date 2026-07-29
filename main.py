"""Mini NPU Simulator — MAC(Multiply-Accumulate) 연산으로 N×N 패턴이
Cross(+) 인지 X 인지 판별하는 콘솔 시뮬레이터.

제약: 표준 라이브러리(json, time 등)만 사용. NumPy 등 외부 라이브러리 금지.
      MAC 연산은 반복문으로 직접 구현한다.
실행: python main.py
"""

import json
import math
import os
import sys
import time

# Windows 콘솔(cp949)에서도 한글·기호(✓, ×)가 깨지지 않도록 UTF-8로 고정.
for _stream in (sys.stdout, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8")   # Python 3.7+
    except (AttributeError, ValueError):
        pass

# --- 정책 상수 -------------------------------------------------------------
EPS = 1e-9              # 점수 동점 판정 허용오차(부동소수점 오차 흡수)
REPEAT = 10            # 성능 측정 반복 횟수(각 크기별 평균)
CROSS, X, UNDECIDED = "Cross", "X", "UNDECIDED"
DATA_FILE = "data.json"


# =========================================================================
# 1. 데이터 구조 — n×n 2차원 값을 저장하고 위치별로 읽고 쓴다.
# =========================================================================
class Matrix:
    """정사각(n×n) 2차원 배열 래퍼. 값 저장/조회 + 1D 평탄화 지원."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.n = len(self.rows)
        # 정사각 검증: 모든 행의 길이가 n 이어야 한다.
        for r in self.rows:
            if len(r) != self.n:
                raise ValueError("정사각(n×n) 행렬이 아닙니다: 행/열 개수 불일치")

    def get(self, r, c):
        return self.rows[r][c]

    def set(self, r, c, value):
        self.rows[r][c] = value

    @property
    def size(self):
        return self.n

    def to_flat(self):
        """2차원 → 1차원(길이 N²) 변환. 보너스(메모리 접근 최적화)에 사용."""
        flat = []
        for row in self.rows:
            flat.extend(row)
        return flat


# =========================================================================
# 2. 라벨 정규화 — 서로 다른 표기를 표준 라벨(Cross / X)로 통일한다.
# =========================================================================
def normalize_expected(symbol):
    """expected 기호를 표준 라벨로. '+' → Cross, 'x'/'X' → X."""
    s = str(symbol).strip().lower()
    if s == "+":
        return CROSS
    if s == "x":
        return X
    raise ValueError("알 수 없는 expected 값: %r" % symbol)


def normalize_filter_label(key):
    """필터 키를 표준 라벨로. 'cross' → Cross, 'x' → X."""
    s = str(key).strip().lower()
    if s == "cross":
        return CROSS
    if s == "x":
        return X
    raise ValueError("알 수 없는 필터 라벨: %r" % key)


# =========================================================================
# 3. MAC 연산 — 같은 위치끼리 곱하고(Multiply) 전부 더한다(Accumulate).
# =========================================================================
def mac_2d(pattern, filt):
    """2차원 반복문으로 직접 구현한 MAC. 크기 불일치는 ValueError."""
    if pattern.size != filt.size:
        raise ValueError(
            "크기 불일치: 패턴 %dx%d vs 필터 %dx%d"
            % (pattern.size, pattern.size, filt.size, filt.size)
        )
    n = pattern.size
    score = 0.0
    for i in range(n):
        prow, frow = pattern.rows[i], filt.rows[i]
        for j in range(n):
            score += prow[j] * frow[j]   # 곱하고 누적
    return score


def mac_1d(flat_pattern, flat_filter):
    """1차원 평탄화 버전 MAC(보너스: 접근 패턴 단순화)."""
    if len(flat_pattern) != len(flat_filter):
        raise ValueError("크기 불일치(1D): %d vs %d" % (len(flat_pattern), len(flat_filter)))
    score = 0.0
    for k in range(len(flat_pattern)):
        score += flat_pattern[k] * flat_filter[k]
    return score


# =========================================================================
# 4. 판정 — 두 점수를 epsilon 기반으로 비교한다.
# =========================================================================
def judge(score_a, score_b, label_a, label_b):
    """|a-b| < EPS 이면 UNDECIDED(동점), 아니면 큰 쪽의 라벨을 반환."""
    if abs(score_a - score_b) < EPS:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


# =========================================================================
# 5. 성능 측정 — I/O를 제외하고 연산 함수 호출 구간만 잰다.
# =========================================================================
def measure_ms(func, repeat=REPEAT):
    """func()를 repeat회 호출한 평균 실행 시간을 ms로 반환."""
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        total += time.perf_counter() - start
    return (total / repeat) * 1000.0   # 초 → ms


# =========================================================================
# 6. 패턴 생성기(보너스) — 크기 N의 Cross/X 패턴을 자동 생성한다.
# =========================================================================
def generate_cross(n):
    """가운데 행/열이 1인 십자가(+) 패턴."""
    mid = n // 2
    return Matrix([[1.0 if (i == mid or j == mid) else 0.0
                    for j in range(n)] for i in range(n)])


def generate_x(n):
    """두 대각선이 1인 X 패턴."""
    return Matrix([[1.0 if (i == j or i + j == n - 1) else 0.0
                    for j in range(n)] for i in range(n)])


# =========================================================================
# 7. 모드1 입력 — 한 줄씩(공백 구분) 받아 검증하고 재입력을 유도한다.
# =========================================================================
def read_matrix(name, n):
    """n줄을 입력받아 n×n Matrix로. 형식 오류 시 안내 후 처음부터 재입력."""
    print("%s (%d줄 입력, 공백 구분)" % (name, n))
    while True:
        rows = []
        ok = True
        for i in range(n):
            try:
                line = input()
            except EOFError:
                raise SystemExit("\n입력이 종료되었습니다.")
            parts = line.split()
            if len(parts) != n:
                print("입력 형식 오류: 각 줄에 %d개의 숫자를 공백으로 구분해 입력하세요. (다시 입력)" % n)
                ok = False
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                print("입력 형식 오류: 숫자만 입력할 수 있습니다. (다시 입력)")
                ok = False
                break
        if ok and len(rows) == n:
            return Matrix(rows)
        # 실패하면 while 루프 처음으로 돌아가 전체 재입력


# =========================================================================
# 8. 숫자 출력 포맷
# =========================================================================
def fmt_score(x):
    """정수형 실수는 깔끔하게, 그 외에는 부동소수점을 그대로 보여 준다.
    inf/nan(예: JSON의 1e309) 은 int() 변환이 불가하므로 그대로 표기한다."""
    if not math.isfinite(x):
        return repr(x)   # inf / -inf / nan
    if x == int(x):
        return "%.1f" % x
    return repr(x)   # 예: 0.8999999999999999 — 부동소수점 오차를 그대로 노출


# =========================================================================
# 9. 모드1 실행 (사용자 입력, 3×3)
# =========================================================================
def run_mode1():
    n = 3
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a = read_matrix("필터 A", n)
    filter_b = read_matrix("필터 B", n)
    print("필터 A, B 저장 완료.")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern = read_matrix("패턴", n)

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    score_a = mac_2d(pattern, filter_a)
    score_b = mac_2d(pattern, filter_b)
    avg_ms = measure_ms(lambda: mac_2d(pattern, filter_a))
    verdict = judge(score_a, score_b, "A", "B")

    print("A 점수: %s" % fmt_score(score_a))
    print("B 점수: %s" % fmt_score(score_b))
    print("연산 시간(평균/%d회): %.4f ms" % (REPEAT, avg_ms))
    if verdict == UNDECIDED:
        print("판정: 판정 불가 (|A-B| < %g)" % EPS)
    else:
        print("판정: %s" % verdict)

    print("\n#----------------------------------------")
    print("# [4] 성능 분석 (3×3)")
    print("#----------------------------------------")
    performance_table([3])


# =========================================================================
# 10. data.json 로드 및 스키마 검증
# =========================================================================
def load_filters(raw_filters):
    """filters 를 {size(int): {Cross: Matrix, X: Matrix}} 로 정규화.
    개별 필터가 깨져 있어도 예외를 던지지 않고 건너뛴다(프로그램 중단 방지)."""
    result = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 형식이 올바르지 않습니다(딕셔너리 아님).")
        return result
    for size_key, labels in raw_filters.items():
        try:
            n = int(str(size_key).split("_")[1])          # 'size_5' → 5
            entry = {}
            for label_key, grid in labels.items():
                std = normalize_filter_label(label_key)   # 'cross' → Cross
                entry[std] = Matrix(grid)
            result[n] = entry
            print("✓ size_%d  필터 로드 완료 (Cross, X)" % n)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
            print("✗ 필터 로드 실패(%s): %s" % (size_key, exc))
    return result


def sort_pattern_key(key):
    """'size_13_2' → (13, 2) 로 자연 정렬. 형식이 안 맞는 키는 뒤로 밀되 크래시하지 않는다."""
    parts = str(key).split("_")
    try:
        return (0, int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return (1, 0, 0)


# =========================================================================
# 11. 성능 분석 표 (크기 / 평균시간 / 연산횟수 N²)
# =========================================================================
def performance_table(sizes):
    print("크기       평균 시간(ms)    연산 횟수(N^2)")
    print("--------------------------------------------")
    for n in sizes:
        pattern = generate_x(n)
        filt = generate_cross(n)
        avg_ms = measure_ms(lambda: mac_2d(pattern, filt))
        label = "%d×%d" % (n, n)
        print("%-10s %-14.4f %d" % (label, avg_ms, n * n))


def bonus_compare_1d_2d(n=25):
    """보너스: 동일 입력에 대해 2D vs 1D(평탄화) MAC 성능 비교."""
    pattern = generate_x(n)
    filt = generate_cross(n)
    flat_p, flat_f = pattern.to_flat(), filt.to_flat()
    ms_2d = measure_ms(lambda: mac_2d(pattern, filt))
    ms_1d = measure_ms(lambda: mac_1d(flat_p, flat_f))
    print("\n#----------------------------------------")
    print("# [보너스] 2D vs 1D(평탄화) 성능 비교 (%d×%d, %d회 평균)" % (n, n, REPEAT))
    print("#----------------------------------------")
    print("2D 접근: %.4f ms" % ms_2d)
    print("1D 접근: %.4f ms" % ms_1d)


# =========================================================================
# 12. 모드2 실행 (data.json 분석)
# =========================================================================
def run_mode2(path=DATA_FILE):
    if not os.path.exists(path):
        print("data.json 을 찾을 수 없습니다: %s" % os.path.abspath(path))
        return
    # 파일 자체가 깨져 읽을 수 없으면(잘못된 JSON 등) 트레이스백 없이 안내하고 종료.
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:   # ValueError 가 json.JSONDecodeError 포함
        print("data.json 을 읽을 수 없습니다: %s" % exc)
        return
    if not isinstance(data, dict):
        print("data.json 최상위 구조가 올바르지 않습니다(객체가 아님).")
        return

    print("\n#----------------------------------------")
    print("# [1] 필터 로드")
    print("#----------------------------------------")
    filters = load_filters(data.get("filters", {}))   # 절대 예외를 던지지 않음

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#----------------------------------------")
    patterns = data.get("patterns", {})
    if not isinstance(patterns, dict):
        patterns = {}
    total = pass_count = 0
    failures = []   # (key, reason)

    for key in sorted(patterns, key=sort_pattern_key):
        total += 1
        item = patterns[key]
        print("--- %s ---" % key)
        try:
            n = int(key.split("_")[1])
            pattern = Matrix(item["input"])
            expected = normalize_expected(item["expected"])

            if n not in filters:
                raise ValueError("size_%d 필터가 없습니다" % n)
            if pattern.size != n:
                raise ValueError("패턴 크기(%d)가 키의 크기(%d)와 다릅니다" % (pattern.size, n))

            cross_filt, x_filt = filters[n][CROSS], filters[n][X]
            score_cross = mac_2d(pattern, cross_filt)   # 크기 불일치면 여기서 ValueError
            score_x = mac_2d(pattern, x_filt)
            verdict = judge(score_cross, score_x, CROSS, X)

            print("Cross 점수: %s" % fmt_score(score_cross))
            print("X 점수: %s" % fmt_score(score_x))
            if verdict == UNDECIDED:
                result = "FAIL"
                reason = "동점(UNDECIDED) 처리 규칙 — |Cross-X| < %g" % EPS
                failures.append((key, reason))
                print("판정: UNDECIDED | expected: %s | FAIL (동점 규칙)" % expected)
            elif verdict == expected:
                result = "PASS"
                pass_count += 1
                print("판정: %s | expected: %s | PASS" % (verdict, expected))
            else:
                result = "FAIL"
                reason = "판정(%s) != expected(%s)" % (verdict, expected)
                failures.append((key, reason))
                print("판정: %s | expected: %s | FAIL" % (verdict, expected))
        except (KeyError, ValueError, TypeError, IndexError, OverflowError) as exc:
            # 스키마/크기/타입/수치 범위 문제 등 — 프로그램을 멈추지 않고 케이스 단위 FAIL.
            reason = "데이터/스키마 오류: %s" % exc
            failures.append((key, reason))
            print("FAIL (%s)" % reason)

    print("\n#----------------------------------------")
    print("# [3] 성능 분석 (평균/%d회)" % REPEAT)
    print("#----------------------------------------")
    performance_table([3, 5, 13, 25])
    bonus_compare_1d_2d(25)

    print("\n#----------------------------------------")
    print("# [4] 결과 요약")
    print("#----------------------------------------")
    fail_count = total - pass_count
    print("총 테스트: %d개" % total)
    print("통과: %d개" % pass_count)
    print("실패: %d개" % fail_count)
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print("- %s: %s" % (key, reason))


# =========================================================================
# 13. 메인 메뉴
# =========================================================================
def main():
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    try:
        choice = input("선택: ").strip()
    except EOFError:
        raise SystemExit("입력이 종료되었습니다.")

    if choice == "1":
        run_mode1()
    elif choice == "2":
        run_mode2()
    else:
        print("잘못된 선택입니다. 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    main()
