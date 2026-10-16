#!/usr/bin/env python3
"""Generate Gemma persona responses and optionally submit them to a Google Form.

The API key is read from the environment or the adjacent .env file. Generated
rows and progress are saved locally so a long batch run can be resumed without
regenerating/submitting rows.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

import form_bot


DEFAULT_BASE_URL = "http://171.226.10.121:8000/llm/v1/"
DEFAULT_MODEL = "gemma-4"
DEFAULT_OUTPUT = Path("gemma_responses.csv")
DEFAULT_STATE = Path(".gemma_form_state.json")
PROMPT_VERSION = "2026-10-16-v1"


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[len("export "):].lstrip()
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if os.environ.get(name, "").strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[name] = value


PERSONA_ARCHETYPES = [
    "ưu tiên môi trường và sức khỏe, dùng không gian xanh thường xuyên",
    "thực tế, cân nhắc rõ lợi ích với khoản phí hàng tháng",
    "quan tâm minh bạch quỹ và mức độ đáng tin của ban quản lý",
    "gia đình có nhu cầu sinh hoạt ngoài trời, coi trọng tiện ích cho trẻ em",
    "bận rộn, dùng không gian xanh không đều và đánh giá vừa phải",
    "nhạy cảm với chi phí, muốn nâng cấp nhưng còn cân nhắc khả năng chi trả",
    "tin vào lợi ích dài hạn và ủng hộ cải thiện cảnh quan nếu được giám sát",
    "ưu tiên chất lượng sống nhưng cần thấy hiệu quả cụ thể trước khi đóng thêm",
    "mới chuyển đến, trải nghiệm còn hạn chế nên có một số đánh giá trung lập",
    "thích không gian xanh, đồng thời có quan điểm độc lập về từng hạng mục",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_state(path: Path, form_url: str, model: str, count: int, batch_size: int) -> dict:
    if not path.exists():
        return {
            "form_url": form_url,
            "model": model,
            "count": count,
            "batch_size": batch_size,
            "prompt_version": PROMPT_VERSION,
            "data_origin": "Gemma-generated synthetic personas for internal benchmark",
            "created_at": iso_time(utc_now()),
            "responses": [],
            "submitted": [],
            "in_flight": None,
            "next_batch_at": None,
        }

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Không đọc được checkpoint {path}: {error}") from error

    expected = {
        "form_url": form_url,
        "model": model,
        "count": count,
        "batch_size": batch_size,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(
                f"Checkpoint {path} có {key}={state.get(key)!r}, nhưng lần chạy này dùng {value!r}. "
                "Dùng cùng tham số để tiếp tục hoặc đổi tên checkpoint để bắt đầu một lượt mới."
            )
    state.setdefault("responses", [])
    state.setdefault("submitted", [])
    state.setdefault("in_flight", None)
    state.setdefault("next_batch_at", None)
    state.setdefault("next_submit_at", None)
    return state


def save_state(path: Path, state: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def api_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise RuntimeError("GEMMA_BASE_URL phải là URL http(s) hợp lệ.")
    return urljoin(base_url.rstrip("/") + "/", "chat/completions")


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemma không trả về JSON object.")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Phản hồi Gemma không phải JSON object.")
    return value


def call_gemma(base_url: str, model: str, api_key: str, user_prompt: str, timeout: int) -> dict:
    endpoint = api_endpoint(base_url)
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn đang tham gia một benchmark nội bộ về cách LLM trả lời khảo sát "
                        "theo persona giả lập. Hãy tạo câu trả lời nhất quán, tự nhiên và chỉ "
                        "dựa trên persona giả định; không tuyên bố đó là người thật."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemma API trả HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Không kết nối được Gemma API: {error.reason}") from error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Phản hồi Gemma API thiếu choices[0].message.content.") from error
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return extract_json(str(content))


def form_signature(form) -> list[tuple[str, int, tuple[str, ...]]]:
    return [
        (field.header, field.entry_id, tuple(field.options))
        for field in form.fields
    ]


def response_schema(form) -> list[dict]:
    return [
        {
            "key": f"entry.{field.entry_id}",
            "question": field.header,
            "type": form_bot.TYPE_NAMES.get(field.kind, str(field.kind)),
            "required": field.required,
            "allowed_answers": field.options,
        }
        for field in form.fields
    ]


def make_prompt(
    form,
    response_number: int,
    correction: str | None = None,
    previous_profile_summaries: list[str] | None = None,
) -> str:
    archetype = PERSONA_ARCHETYPES[(response_number - 1) % len(PERSONA_ARCHETYPES)]
    likert_fields = sum(1 for field in form.fields if field.kind == form_bot.GRID)
    schema_json = json.dumps(response_schema(form), ensure_ascii=False, indent=2)
    required_fields = [field for field in form.fields if field.required]
    required_keys = ", ".join(
        f"entry.{field.entry_id}" for field in required_fields
    ) or "(không có câu bắt buộc)"
    previous = previous_profile_summaries or []
    previous_text = "\n".join(f"- {summary}" for summary in previous[-10:]) or "(chưa có)"
    correction_text = f"\nSửa các lỗi định dạng sau: {correction}\n" if correction else ""

    return f"""Tạo phản hồi số {response_number} trong một benchmark nội bộ về cách Gemma trả lời khảo sát theo persona.

Persona giả lập lần này có xu hướng: {archetype}.
Hãy tự chọn một hồ sơ giả định hợp lý của cư dân Việt Nam tại Vinhomes Smart City: tuổi trưởng thành trong khoảng 20–70, phân khu, giới tính theo lựa chọn có sẵn, học vấn, hình thức cư trú và tần suất sử dụng không gian xanh phải ăn khớp với nhau. Hồ sơ là giả định cho thí nghiệm, không đại diện cho một cá nhân thật. Biểu mẫu này không có câu hỏi tên; không tạo thêm tên.

Hướng dẫn cho các câu Likert (có {likert_fields} dòng): ưu tiên điểm 3–5. Chỉ dùng 1–2 ở một số ít mục khi thật sự phù hợp với hồ sơ; không ép mọi câu thành điểm cao. Các đánh giá phải nhất quán với tính cách, khả năng chi trả, mức sử dụng không gian xanh và niềm tin vào ban quản lý. Tránh chuỗi điểm lặp, tăng/giảm đều hoặc mẫu luân phiên máy móc; không trả lời zic-zac.

Trả về duy nhất một JSON object theo cấu trúc:
{{"persona_summary":"mô tả ngắn hồ sơ giả lập","answers":{{"entry.ID":"câu trả lời"}}}}

Với lựa chọn trắc nghiệm, dùng nguyên văn một giá trị trong allowed_answers. Với câu hỏi lưới, dùng chính xác chuỗi "1", "2", "3", "4" hoặc "5". Với câu tuổi trả lời chỉ bằng số tuổi. Trong answers, dùng đúng key entry.ID và phải có đủ mọi key bắt buộc dưới đây với giá trị không rỗng. Trước khi trả lời, tự đối chiếu rằng không thiếu key nào. Không thêm lời giải thích ngoài JSON.

Danh sách key bắt buộc:
{required_keys}

Các hồ sơ gần nhất để tránh lặp lại: 
{previous_text}

Schema câu hỏi:
{schema_json}
{correction_text}"""


def normalize_model_answers(form, result: dict) -> tuple[dict[str, str], str]:
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("JSON Gemma thiếu object 'answers'.")

    summary = str(result.get("persona_summary", "Persona Gemma giả lập")).strip()
    row: dict[str, str] = {}
    for field in form.fields:
        key = f"entry.{field.entry_id}"
        value = answers.get(key)
        if value is None:
            value = answers.get(field.header)
        if isinstance(value, (int, float)):
            value = str(value)
        elif value is not None:
            value = str(value).strip()
        row[field.header] = value or ""
        field.column = field.header

    _, errors = form_bot.encode_row(form, row)
    likert = [
        row[field.header]
        for field in form.fields
        if field.kind == form_bot.GRID and field.options == ["1", "2", "3", "4", "5"]
    ]
    low_scores = sum(value in ("1", "2") for value in likert)
    low_score_limit = max(1, int(len(likert) * 0.15 + 0.999))
    if low_scores > low_score_limit:
        errors.append(
            f"có {low_scores} điểm 1–2, vượt giới hạn {low_score_limit} cho persona này"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return row, summary


def generate_response(
    form,
    response_number: int,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    previous_profiles: list[str],
) -> tuple[dict[str, str], str]:
    correction = None
    max_attempts = 4
    for attempt in range(max_attempts):
        result = call_gemma(
            base_url,
            model,
            api_key,
            make_prompt(form, response_number, correction, previous_profiles),
            timeout,
        )
        try:
            return normalize_model_answers(form, result)
        except ValueError as error:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Gemma vẫn trả lời sai định dạng sau {max_attempts} lần: {error}"
                ) from error
            correction = (
                f"{error}. Hãy tạo lại đầy đủ object answers; đặc biệt không được bỏ sót "
                "bất kỳ key bắt buộc nào trong danh sách của prompt."
            )
    raise RuntimeError("Không tạo được phản hồi hợp lệ.")


def write_csv(path: Path, form, responses: list[dict]) -> None:
    import csv

    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.header for field in form.fields])
        writer.writeheader()
        for item in responses:
            writer.writerow(item["answers"])
    temp.replace(path)


def ensure_form_unchanged(form_url: str, expected_signature) -> object:
    current = form_bot.load_form(form_url)
    if form_signature(current) != expected_signature:
        raise RuntimeError("Cấu trúc Form đã thay đổi trong lúc chạy; dừng để tránh gửi lệch câu trả lời.")
    return current


def wait_until(when: datetime, what: str = "Đợt kế tiếp") -> None:
    while True:
        remaining = (when - utc_now()).total_seconds()
        if remaining <= 0:
            return
        minutes = max(1, int(remaining // 60))
        print(f"{what} bắt đầu sau khoảng {minutes} phút. Có thể dừng bằng Ctrl+C rồi chạy lại để tiếp tục.")
        time.sleep(min(60, remaining))


def generate_up_to(
    target: int,
    form,
    state: dict,
    state_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
) -> None:
    profiles = [item.get("persona_summary", "") for item in state["responses"]]
    while len(state["responses"]) < target:
        number = len(state["responses"]) + 1
        row, summary = generate_response(
            form,
            number,
            base_url,
            model,
            api_key,
            timeout,
            profiles,
        )
        state["responses"].append(
            {
                "persona_summary": summary,
                "generated_at": iso_time(utc_now()),
                "answers": row,
            }
        )
        profiles.append(summary)
        save_state(state_path, state)
        write_csv(output_path, form, state["responses"])
        print(f"Đã tạo và lưu phản hồi Gemma {number}/{state['count']}.")


def run(args) -> int:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    api_key = os.environ.get("GEMMA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Chưa có GEMMA_API_KEY trong .env hoặc biến môi trường.")

    form = form_bot.load_form(args.form_url)
    if not form.fields:
        raise RuntimeError("Không tìm thấy câu hỏi nào trong Google Form.")
    expected_signature = form_signature(form)
    state = load_state(args.state, args.form_url, args.model, args.count, args.batch_size)

    if state.get("in_flight") is not None:
        idx = state["in_flight"]
        raise RuntimeError(
            f"Checkpoint báo phản hồi {idx + 1} đang ở trạng thái chưa xác định sau lần gửi trước. "
            "Kiểm tra Google Forms → Câu trả lời trước; không tự chạy lại để tránh gửi trùng."
        )

    if not args.submit:
        generate_up_to(
            args.count,
            form,
            state,
            args.state,
            args.output,
            args.base_url,
            args.model,
            api_key,
            args.timeout,
        )
        print(f"Đã lưu {len(state['responses'])} phản hồi vào {args.output}; chưa gửi Form.")
        return 0

    while len(state["submitted"]) < args.count:
        next_batch_at = parse_time(state.get("next_batch_at"))
        if next_batch_at and next_batch_at > utc_now():
            wait_until(next_batch_at)
        state["next_batch_at"] = None
        save_state(args.state, state)

        pending = [i for i in range(args.count) if i not in state["submitted"]]
        batch = pending[: args.batch_size]
        target = max(batch) + 1
        generate_up_to(
            target,
            form,
            state,
            args.state,
            args.output,
            args.base_url,
            args.model,
            api_key,
            args.timeout,
        )

        for position, index in enumerate(batch, start=1):
            next_submit_at = parse_time(state.get("next_submit_at"))
            if next_submit_at and next_submit_at > utc_now():
                wait_until(next_submit_at, "Lần gửi kế tiếp")
            live_form = ensure_form_unchanged(args.form_url, expected_signature)
            for field in live_form.fields:
                field.column = field.header
            row = state["responses"][index]["answers"]
            pairs, errors = form_bot.encode_row(live_form, row)
            if errors:
                raise RuntimeError(f"Phản hồi {index + 1} không hợp lệ: " + "; ".join(errors))

            state["in_flight"] = index
            state["last_submit_started_at"] = iso_time(utc_now())
            save_state(args.state, state)
            success, detail = form_bot.submit(live_form, pairs, retries=0)
            if not success:
                state["last_submit_error"] = detail
                save_state(args.state, state)
                raise RuntimeError(
                    f"Không xác nhận được phản hồi {index + 1} ( {detail} ). "
                    "Đã giữ checkpoint để bạn kiểm tra thủ công trước khi tiếp tục."
                )

            state["submitted"].append(index)
            state["submitted"].sort()
            state["in_flight"] = None
            state["last_submit_error"] = None
            state["last_batch_completed_at"] = iso_time(utc_now())
            delay_minutes = None
            if position < len(batch):
                delay_minutes = random.uniform(args.min_row_delay_minutes, args.max_row_delay_minutes)
                state["next_submit_at"] = iso_time(utc_now() + timedelta(minutes=delay_minutes))
            else:
                state["next_submit_at"] = None
            save_state(args.state, state)
            print(f"Đã gửi đợt: phản hồi {index + 1}/{args.count} ({position}/{len(batch)} trong đợt).")
            if delay_minutes is not None:
                print(f"Nghỉ ngẫu nhiên {delay_minutes:.1f} phút trước lần gửi kế tiếp.")

        remaining = args.count - len(state["submitted"])
        if remaining:
            pause_hours = random.uniform(args.min_pause_hours, args.max_pause_hours)
            next_at = utc_now() + timedelta(hours=pause_hours)
            state["next_batch_at"] = iso_time(next_at)
            save_state(args.state, state)
            print(f"Đã xong đợt {len(state['submitted']) // args.batch_size}. Nghỉ {pause_hours:.2f} giờ trước đợt kế tiếp.")

    print(f"Hoàn tất: đã gửi {len(state['submitted'])} phản hồi. Checkpoint: {args.state}.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Dùng Gemma tạo persona trả lời khảo sát và tùy chọn gửi theo từng đợt."
    )
    parser.add_argument("form_url", help="Link Google Form có quyền nhận phản hồi")
    parser.add_argument("--count", type=int, default=50, help="Số persona/phản hồi (mặc định 50)")
    parser.add_argument("--batch-size", type=int, default=10, help="Số phản hồi mỗi đợt (mặc định 10)")
    parser.add_argument("--min-pause-hours", type=float, default=2.0, help="Thời gian nghỉ tối thiểu giữa các đợt")
    parser.add_argument("--max-pause-hours", type=float, default=3.0, help="Thời gian nghỉ tối đa giữa các đợt")
    parser.add_argument("--min-row-delay-minutes", type=float, default=3.0, help="Số phút nghỉ tối thiểu giữa hai lần gửi trong cùng đợt")
    parser.add_argument("--max-row-delay-minutes", type=float, default=10.0, help="Số phút nghỉ tối đa giữa hai lần gửi trong cùng đợt")
    parser.add_argument("--base-url", default=os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=90, help="Timeout mỗi yêu cầu API, tính bằng giây")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV câu trả lời được lưu cục bộ")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Checkpoint để tiếp tục sau khi dừng")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Gửi phản hồi lên Google Form. Nếu bỏ cờ này, chỉ tạo CSV và không gửi.",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count phải ít nhất là 1")
    if args.batch_size < 1:
        parser.error("--batch-size phải ít nhất là 1")
    if args.min_pause_hours < 0 or args.max_pause_hours < args.min_pause_hours:
        parser.error("Khoảng --min-pause-hours/--max-pause-hours không hợp lệ")
    if args.min_row_delay_minutes < 0 or args.max_row_delay_minutes < args.min_row_delay_minutes:
        parser.error("Khoảng --min-row-delay-minutes/--max-row-delay-minutes không hợp lệ")
    if args.timeout < 1:
        parser.error("--timeout phải dương")
    return args


def main(argv=None) -> int:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Đã dừng. Checkpoint được giữ lại; chạy lại đúng lệnh cũ để tiếp tục.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, form_bot.BotError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
