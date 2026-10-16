#!/usr/bin/env python3
"""Bot điền Google Form hàng loạt từ file CSV hoặc Google Sheet.

    python3 form_bot.py inspect FORM_URL [--template mau.csv]
    python3 form_bot.py fill FORM_URL DU_LIEU [--dry-run] [--limit N]

Chỉ dùng thư viện chuẩn của Python (>= 3.9).
"""

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (compatible; form-bot/1.0)"
STATE_FILE = Path(".form_bot_state.json")

# Mã loại câu hỏi trong FB_PUBLIC_LOAD_DATA_ của Google Form
SHORT, PARAGRAPH, CHOICE, DROPDOWN, CHECKBOX, SCALE, GRID, PAGE_BREAK, DATE, TIME, UPLOAD = (
    0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 13,
)
TYPE_NAMES = {
    SHORT: "Trả lời ngắn", PARAGRAPH: "Đoạn văn", CHOICE: "Trắc nghiệm",
    DROPDOWN: "Menu thả xuống", CHECKBOX: "Hộp kiểm", SCALE: "Thang tuyến tính",
    GRID: "Lưới", DATE: "Ngày", TIME: "Giờ", UPLOAD: "Tải tệp lên",
}
MULTI_SEP = "|"


class BotError(Exception):
    pass


@dataclass
class Field:
    header: str
    entry_id: int
    kind: int
    required: bool
    options: list = field(default_factory=list)
    has_other: bool = False
    column: str = None  # tên cột CSV được ghép với câu hỏi này


@dataclass
class Form:
    title: str
    fields: list
    pages: int
    post_url: str
    hidden: dict


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def http(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=30)


# --------------------------------------------------------------------------- form


def load_form(form_url):
    try:
        with http(form_url) as resp:
            final_url = resp.geturl()
            page = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise BotError(f"Không mở được form (HTTP {e.code}). Kiểm tra lại URL.") from e
    except urllib.error.URLError as e:
        raise BotError(f"Lỗi mạng khi mở form: {e.reason}") from e

    if "accounts.google.com" in final_url:
        raise BotError("Form yêu cầu đăng nhập Google — bot không điền được loại form này.")
    m = re.search(r"FB_PUBLIC_LOAD_DATA_\s*=\s*(.*?);\s*</script>", page, re.S)
    if not m:
        raise BotError(
            "Không đọc được cấu trúc form. Hãy dùng link dạng "
            "https://docs.google.com/forms/d/e/.../viewform (nút 'Gửi' > link), "
            "không dùng link chỉnh sửa (/edit)."
        )
    data = json.loads(m.group(1))

    fields, pages = [], 1
    for item in data[1][1] or []:
        kind = item[3]
        if kind == PAGE_BREAK:
            pages += 1
            continue
        if kind not in TYPE_NAMES or len(item) < 5 or not item[4]:
            continue
        title = (item[1] or "").strip()
        for q in item[4]:
            opts = q[1] or []
            header = title
            if kind == GRID:
                header = f"{title} [{q[3][0]}]"
            fields.append(Field(
                header=header,
                entry_id=q[0],
                kind=kind,
                required=bool(q[2]),
                options=[o[0] for o in opts if o and o[0]],
                has_other=any(len(o) > 4 and o[4] for o in opts),
            ))

    hidden = {}
    for tag in re.findall(r"<input[^>]*type=\"hidden\"[^>]*>", page):
        name = re.search(r'name="([^"]*)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            hidden[name.group(1)] = html.unescape(value.group(1)) if value else ""
    hidden["pageHistory"] = ",".join(str(i) for i in range(pages))

    base = final_url.split("?")[0]
    if base.endswith("/viewform"):
        base = base[: -len("viewform")] + "formResponse"
    elif not base.endswith("/formResponse"):
        raise BotError(f"URL form không nhận dạng được: {final_url}")

    title = data[3] if len(data) > 3 and isinstance(data[3], str) else ""
    return Form(title, fields, pages, base, hidden)


# --------------------------------------------------------------------------- data


def sheet_to_csv_url(url):
    m = re.search(r"/spreadsheets/d/([\w-]+)", url)
    if not m:
        return url
    gid = re.search(r"[#&?]gid=(\d+)", url)
    out = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return out + (f"&gid={gid.group(1)}" if gid else "")


def read_rows(source):
    if re.match(r"https?://", source):
        try:
            with http(sheet_to_csv_url(source)) as resp:
                if "accounts.google.com" in resp.geturl():
                    raise BotError(
                        "Google Sheet đang ở chế độ riêng tư. Hãy chia sẻ 'Bất kỳ ai có "
                        "đường liên kết' (người xem) hoặc tải về CSV rồi truyền đường dẫn file."
                    )
                text = resp.read().decode("utf-8-sig")
        except urllib.error.URLError as e:
            raise BotError(f"Không tải được dữ liệu: {e}") from e
    else:
        try:
            text = Path(source).read_text(encoding="utf-8-sig")
        except OSError as e:
            raise BotError(f"Không đọc được file: {e}") from e

    reader = csv.DictReader(io.StringIO(text))
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    return reader.fieldnames or [], rows


def map_columns(form, headers):
    by_norm = {norm(h): h for h in headers}
    for f in form.fields:
        f.column = by_norm.get(norm(f.header)) or by_norm.get(f"entry.{f.entry_id}")
    used = {f.column for f in form.fields}
    return [h for h in headers if h not in used]


# --------------------------------------------------------------------------- encode


def parse_date(text):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def match_option(f, value):
    for opt in f.options:
        if norm(opt) == norm(value):
            return opt
    return None


def encode_row(form, row):
    """Trả về (danh_sách_cặp_key_value, danh_sách_lỗi)."""
    pairs, errors = [], []
    for f in form.fields:
        raw = (row.get(f.column) or "").strip() if f.column else ""
        key = f"entry.{f.entry_id}"
        if not raw:
            if f.required:
                errors.append(f"thiếu giá trị cho câu bắt buộc '{f.header}'")
            continue

        if f.kind == UPLOAD:
            errors.append(f"'{f.header}': câu hỏi tải tệp không được hỗ trợ")
        elif f.kind in (SHORT, PARAGRAPH, SCALE):
            pairs.append((key, raw))
        elif f.kind in (CHOICE, DROPDOWN, CHECKBOX, GRID):
            values = [v.strip() for v in raw.split(MULTI_SEP)] if f.kind in (CHECKBOX, GRID) else [raw]
            for value in filter(None, values):
                opt = match_option(f, value) if f.options else value
                if opt:
                    pairs.append((key, opt))
                elif f.has_other:
                    pairs.append((key, "__other_option__"))
                    pairs.append((f"{key}.other_option_response", value))
                else:
                    errors.append(f"'{f.header}': '{value}' không nằm trong {f.options}")
        elif f.kind == DATE:
            d = parse_date(raw)
            if d is None:
                errors.append(f"'{f.header}': ngày '{raw}' sai định dạng (dùng YYYY-MM-DD hoặc DD/MM/YYYY)")
            else:
                pairs += [(f"{key}_year", str(d.year)), (f"{key}_month", str(d.month)), (f"{key}_day", str(d.day))]
        elif f.kind == TIME:
            m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", raw)
            if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
                errors.append(f"'{f.header}': giờ '{raw}' sai định dạng (dùng HH:MM)")
            else:
                pairs += [(f"{key}_hour", str(int(m.group(1)))), (f"{key}_minute", m.group(2))]
    return pairs, errors


# --------------------------------------------------------------------------- submit


def submit(form, pairs, retries=2):
    body = urllib.parse.urlencode(pairs + list(form.hidden.items())).encode()
    for attempt in range(retries + 1):
        try:
            with http(form.post_url, body) as resp:
                if "accounts.google.com" in resp.geturl():
                    return False, "form yêu cầu đăng nhập"
                return resp.status == 200, f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                return False, f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if attempt == retries:
                return False, f"lỗi mạng: {e.reason}"
        time.sleep(2 ** attempt)
    return False, "không rõ"


def row_key(index, row):
    digest = hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return f"{index}:{digest}"


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------- commands


def cmd_inspect(args):
    form = load_form(args.form_url)
    print(f"Form: {form.title or '(không tên)'} — {form.pages} trang, {len(form.fields)} ô nhập\n")
    for i, f in enumerate(form.fields, 1):
        flags = "bắt buộc" if f.required else "tùy chọn"
        print(f"{i:>2}. {f.header}  [{TYPE_NAMES[f.kind]}, {flags}, entry.{f.entry_id}]")
        if f.options:
            extra = " (+ 'Khác')" if f.has_other else ""
            print(f"      lựa chọn: {' | '.join(f.options)}{extra}")
        if f.kind == CHECKBOX:
            print(f"      nhiều đáp án: ngăn cách bằng '{MULTI_SEP}'")
        if f.kind == UPLOAD:
            print("      (không hỗ trợ)")
    if args.template:
        with open(args.template, "w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerow([f.header for f in form.fields])
        print(f"\nĐã tạo file mẫu: {args.template} (điền dữ liệu từ dòng 2)")


def cmd_fill(args):
    form = load_form(args.form_url)
    headers, rows = read_rows(args.data)
    unused = map_columns(form, headers)

    unmapped = [f for f in form.fields if not f.column and f.kind != UPLOAD]
    for f in unmapped:
        level = "LỖI" if f.required else "cảnh báo"
        print(f"[{level}] không có cột CSV cho câu '{f.header}'")
    if unused:
        print(f"[cảnh báo] cột không khớp câu hỏi nào, bị bỏ qua: {', '.join(unused)}")
    if any(f.required for f in unmapped):
        raise BotError("Thiếu cột cho câu hỏi bắt buộc. Chạy 'inspect --template' để lấy file mẫu.")

    state = load_state()
    done = set(state.get(form.post_url, []))
    selected = list(enumerate(rows, 1))[args.start - 1:]
    if args.limit:
        selected = selected[: args.limit]
    print(f"\n{len(rows)} dòng dữ liệu, xử lý {len(selected)} dòng"
          f"{' (chạy thử, không gửi)' if args.dry_run else ''}.\n")

    ok = failed = skipped = 0
    for n, (index, row) in enumerate(selected):
        label = next((v for v in row.values() if v), "")[:30]
        key = row_key(index, row)
        if key in done and not args.force:
            skipped += 1
            print(f"[{index}] bỏ qua — đã gửi trước đó ({label})")
            continue

        pairs, errors = encode_row(form, row)
        if errors:
            failed += 1
            print(f"[{index}] LỖI dữ liệu ({label}): " + "; ".join(errors))
            continue
        if args.dry_run:
            ok += 1
            print(f"[{index}] hợp lệ ({label}) -> {len(pairs)} trường")
            continue

        success, detail = submit(form, pairs)
        if success:
            ok += 1
            done.add(key)
            state[form.post_url] = sorted(done)
            STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
            print(f"[{index}] đã gửi ({label})")
        else:
            failed += 1
            print(f"[{index}] GỬI THẤT BẠI ({label}): {detail}")
        if n < len(selected) - 1:
            time.sleep(args.delay)

    print(f"\nXong: {ok} {'hợp lệ' if args.dry_run else 'đã gửi'}, {failed} lỗi, {skipped} bỏ qua.")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bot điền Google Form từ CSV / Google Sheet")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="liệt kê các câu hỏi của form")
    p.add_argument("form_url")
    p.add_argument("--template", help="ghi file CSV mẫu (hàng tiêu đề) ra đường dẫn này")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("fill", help="điền form từ dữ liệu")
    p.add_argument("form_url")
    p.add_argument("data", help="file CSV hoặc link Google Sheet (chia sẻ công khai)")
    p.add_argument("--dry-run", action="store_true", help="chỉ kiểm tra dữ liệu, không gửi")
    p.add_argument("--delay", type=float, default=1.0, help="giây nghỉ giữa 2 lần gửi (mặc định 1)")
    p.add_argument("--start", type=int, default=1, help="bắt đầu từ dòng dữ liệu số N (mặc định 1)")
    p.add_argument("--limit", type=int, help="chỉ xử lý N dòng")
    p.add_argument("--force", action="store_true", help="gửi lại cả những dòng đã gửi trước đó")
    p.set_defaults(func=cmd_fill)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except BotError as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
