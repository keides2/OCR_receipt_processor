# -*- coding: utf-8 -*-
from pathlib import Path
import re
import datetime
import shutil
import pandas as pd
from io import BytesIO
from PIL import Image, ImageFilter
from google.cloud import vision

import warnings
# 全ての DeprecationWarning を無視
warnings.simplefilter("ignore", DeprecationWarning)
from pykakasi import kakasi

from rapidfuzz import fuzz
from google.cloud import vision

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
vision_client = vision.ImageAnnotatorClient()
RECEIPT_FOLDER = Path(r"G:\マイドライブ\receipt\receipt_img")
CSV_FOLDER = Path(r"G:\マイドライブ\receipt\csv_statements")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
FUZZ_THRESHOLD = 20
FUZZ_METHODS = [fuzz.partial_ratio, fuzz.token_set_ratio, fuzz.token_sort_ratio]

# キーは正規化＋大文字化した raw_store、値は正しいカタカナ
BRAND_MAP = {
    "MOS BURGER": "モスノネツトチユウモン",
    "KFC": "ケンタツキ-フライドチキン",
    "オンモールサカイテッポウマチ": "イオンモールサカイテッポウチヨウ",
    "コーナン": "コーナン商事",

    # 既存
    "ENEOS":           "エネオス",
    "ENEOS-SS":        "エネオス",
    "ENEJET":          "エネジェット",

    # 追加分
    "LIQUOR HAP":      "リカーハウスエース",
    "リカ-ハウスエ-ス": "リカーハウスエース",

    "BBAR UMEDA":      "バカラパシフィック",

    "SPAGHETTI.":      "ジヨリ-パスタ",
    "SPAGHETTI":       "ジヨリ-パスタ",
    "ジヨリ-パスタ":    "ジヨリ-パスタ",

    "CURRY HOUSE":     "ココイチバンヤ サカイナカモズ",

    "YAKINIKU GEN":    "楽天SP 焼肉玄",
    "ヤキニクゲン":     "楽天SP 焼肉玄",

    "MY SERIA":        "セリア ナカモズテン",
    "MYSERIA":         "セリア ナカモズテン",
    "マイセリア":       "セリア ナカモズテン",

    "FAMILYMART":     "ファミリーマート",
    "—FAMILYMART":     "ファミリーマート",
    "— FAMILYMART":    "ファミリーマート",
}

# kakasi の初期化（引き続き setMode を呼び出しますが警告は出ません）
kks = kakasi()
kks.setMode("J", "K")   # 漢字→カタカナ
kks.setMode("H", "K")   # ひらがな→カタカナ
kks.setMode("a", "K")   # 英字→カタカナ


def to_katakana(s: str) -> str:
    items = kks.convert(s or "")
    return "".join(item["kana"] for item in items)


def normalize_text(s: str) -> str:
    """
    NFKC正規化＋全角スペース→半角＋連続空白→単一空白＋前後トリム

    Args:
        s (str): 入力文字列

    Returns:
        str: 正規化後文字列
    """
    import unicodedata, re
    t = unicodedata.normalize("NFKC", s or "")
    t = t.replace("\u3000", " ")        # 全角スペース→半角
    t = re.sub(r"\s+", " ", t)          # 連続空白→1つ
    return t.strip()


def normalize_spaces(s: str) -> str:
    """
    全角・半角スペース、タブ、改行をすべて除去した文字列を返す
    """
    return re.sub(r"\s+", "", s or "")


def strip_suffix(s: str) -> str:
    return re.sub(r'(マチ|チヨウ|チョウ)$', '', s)


def canonical_kana(s: str) -> str:
    norm = normalize_text(s)
    return re.sub(r'ツ(?=[ア-ヺ])', 'ッ', norm)


def parse_date(s: str) -> datetime.date:
    s = (s or "").replace(" ", "")
    for pat in [r"(\d{4})年(\d{1,2})月(\d{1,2})日",
                r"(\d{4})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})"]:
        m = re.search(pat, s)
        if m:
            y, M, d = m.groups()
            return datetime.date(int(y), int(M), int(d))
    return None


def preprocess_image(path: Path) -> bytes:
    img = Image.open(path).convert("L").filter(ImageFilter.SHARPEN)
    bw  = img.point(lambda x: 0 if x < 140 else 255, '1')
    buf = BytesIO(); bw.save(buf, format="PNG")
    return buf.getvalue()


def extract_receipt_info(image_path: Path) -> dict:
    """
    レシートから
      ・利用日（date_obj: datetime.date）
      ・店舗名（BRAND_MAP適用後のstore_kana）
      ・請求金額（amount: int）
    を抽出して返します。
    """

    # --- OCR 実行 ---
    content = preprocess_image(image_path)
    resp = vision_client.document_text_detection(
        image=vision.Image(content=content),
        image_context={"language_hints": ["ja"]}
    )
    if resp.error.message:
        raise RuntimeError(f"OCRエラー: {resp.error.message}")

    text  = resp.full_text_annotation.text
    lines = text.splitlines()

    # --- 1) 日付抽出（デバッグ付き） ---
    date_obj = None

    # --- 1a-1) 「ご利用日」ラベル以降の全行を順にチェックして日付を探す ---
    for idx, ln in enumerate(lines):
        if "ご利用日" in ln:
            print(f"[DEBUG] ご利用日ラベル行: line[{idx}]={ln!r}")
            for j, nxt in enumerate(lines[idx+1:], start=idx+1):
                print(f"[DEBUG]  ラベル後行 line[{j}]={nxt!r}")
                m = re.match(r"([0-9]{2,4})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})", nxt)
                if not m:
                    continue
                y_t, M_t, d_t = m.groups()
                year  = int(y_t) if len(y_t)==4 else 2000 + int(y_t)
                month = int(M_t); day = int(d_t)
                print(f"[DEBUG]  ラベル後行マッチ groups={m.groups()} → {year}-{month}-{day}")
                if 1 <= month <= 12 and 1 <= day <= 31:
                    date_obj = datetime.date(year, month, day)
                break
            # ラベル以降をスキャンし終えたらループ脱出

    # --- 1a-2) ご利用日パターン全文検索（行内に日付がまとまっている場合） ---
    if date_obj is None:
        print("[DEBUG] トライ: ご利用日パターン")
        m = re.search(
            r"ご利用日[:：]?\s*([0-9]{2,4})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})",
            text
        )
        if m:
            y_t, M_t, d_t = m.groups()
            year  = int(y_t) if len(y_t)==4 else 2000+int(y_t)
            month = int(M_t); day = int(d_t)
            print(f"[DEBUG] ご利用日マッチ: {m.groups()} → {year}-{month}-{day}")
            if 1<=month<=12 and 1<=day<=31:
                date_obj = datetime.date(year, month, day)
        else:
            print("[DEBUG] ご利用日パターンマッチなし")

    # --- 1b) フォールバック：20XX年／20XX/MM/DD --- 
    if date_obj is None:
        for pat in (
            r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日",
            r"(20\d{2})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})"
        ):
            print(f"[DEBUG] トライ: パターン {pat}")
            m = re.search(pat, text)
            if m:
                y, M, d = m.groups()
                month, day = int(M), int(d)
                print(f"[DEBUG] 直マッチ: {m.group(0)} groups={m.groups()}")
                if 1 <= month <= 12 and 1 <= day <= 31:
                    date_obj = datetime.date(int(y), month, day)
                    print(f"[DEBUG] date_obj に変換: {date_obj}")
                    break
                else:
                    print(f"[DEBUG] 範囲外の月日: {month}, {day}")
            else:
                print("[DEBUG] マッチなし")
                
    if date_obj is None:
        print("[DEBUG] 日付検出最終失敗。全文と行を再確認します。")
        for i, ln in enumerate(lines):
            print(f"[DEBUG] line[{i}]: {ln}")
        raise ValueError("日付検出失敗")

    # （必要なら過去年補正）
    today = datetime.date.today()
    if date_obj.year < today.year - 1:
        try:
            cand = date_obj.replace(year=date_obj.year + 10)
            if cand <= today:
                date_obj = cand
        except ValueError:
            pass

    # --- 2) 金額抽出 ---
    amount = None
    for ln in lines:
        key = normalize_spaces(ln)
        # print(f"[DEBUG] 金額行候補 key='{key}'  line='{ln.strip()}'")
        # 「合計」を含み「小計」を含まない行を探す
        if "合計" in key and not key.startswith("小計"):
            m = re.search(r"[¥￥\\]\s*([\d,]+)", ln)
            if m:
                amount = int(m.group(1).replace(",", ""))
                # print(f"[DEBUG] 抽出金額 → {amount}")
                break

    if amount is None:
        # フォールバック：全文から数値を全部抽出
        nums = re.findall(r"[¥￥\\]\s*([\d,]+)", text)
        if not nums:
            raise ValueError("金額検出失敗")
        from collections import Counter
        # 出現頻度の最も高いものを優先
        ctr = Counter(nums)
        most, freq = ctr.most_common(1)[0]
        if freq > 1:
            amount = int(most.replace(",", ""))
        else:
            # 頻度が1回だけなら従来どおり最大値
            amount = max(int(n.replace(",", "")) for n in nums)

    # --- 3) 店舗名抽出 ---
    m_store = re.search(r"加盟店名\s*([ァ-ヴー\dA-Za-z\-\—\s]+)", text)
    if m_store:
        raw_store = m_store.group(1).strip()
    else:
        raw_store = ""
        for ln in lines:
            if re.search(r"[一-龯ぁ-んァ-ヴーA-Za-z]", ln) and len(ln.strip()) > 1:
                raw_store = ln.strip()
                break
        if not raw_store and lines:
            raw_store = lines[0].strip()

    # --- 4) 店舗名マッピング ---
    eng = normalize_spaces(raw_store).upper()
    eng = re.sub(r'(?:-SS|/NFC)$', '', eng)
    if eng in BRAND_MAP:
        store_kana = BRAND_MAP[eng]
    else:
        kana_tmp = canonical_kana(to_katakana(raw_store))
        key      = normalize_spaces(kana_tmp).upper()
        key      = re.sub(r'(?:-SS|/NFC)$', '', key)
        store_kana = BRAND_MAP.get(key, kana_tmp)

    return {
        "date_obj":   date_obj,
        "raw_store":  raw_store,
        "store_kana": store_kana,
        "amount":     amount
    }


def mark_receipt_in_csv(csv_path: Path, info: dict, mismatches: list) -> bool:
    """
    CSVを読み込み、OCR結果(info)の日付・金額と照合し、
    レシートチェック列を更新して上書き保存します。

    • OCRで「リョウシュウ（領収書誤認）」と読まれた場合は自動一致を行わず、
      正しいOCRを別途試行できるようスキップします。
    • OCRで「ライフ」のように短い店名の場合は、
      CSVの「ライフナカモズテン」等、接頭辞一致を優先してマッチさせます。
    • それ以外は従来どおり英字→BRAND_MAP／カタカナ→BRAND_MAP→
      部分文字列一致（RapidFuzz）でマッチングします。

    Args:
        csv_path (Path): CSVファイルパス
        info (dict): extract_receipt_info の戻り値
        mismatches (list): 不一致パターン記録用リスト

    Returns:
        bool: 一致した行があれば True、なければ False
    """
    import pandas as pd
    import re
    from rapidfuzz import fuzz

    # OCR結果
    ocr_date = info["date_obj"]
    ocr_amt  = info["amount"]
    ocr_kana = info["store_kana"]

    # 「リョウシュウ」など誤認識とみなすキーワード
    MISREAD = {"リョウシュウ"}

    # 誤認識ならスキップ
    if ocr_kana in MISREAD:
        print(f"[DEBUG] OCR店名が誤認識と判断、マッチングをスキップ: {ocr_kana}")
        return False

    # CSV 読み込み
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp932", dtype=str)

    # 列検出
    date_col   = next((c for c in df.columns if c in ["利用日","取引日","取引年月日"]), None)
    store_col  = next((c for c in df.columns if c in ["利用店名","利用店名・商品名","店舗名","加盟店名"]), None)
    amount_col = next((c for c in df.columns if c in ["利用金額","金額","取引金額"]), None)

    # チェック列追加
    if "レシートチェック" not in df.columns:
        df["レシートチェック"] = False

    matched = False
    for idx, row in df.iterrows():
        # 日付・金額の照合（±5日以内を許容）
        # 日付文字列をパース
        d_obj = parse_date(str(row[date_col]))
        # 失敗時はスキップ
        if d_obj is None:
            continue
        # 金額パース
        try:
            amt = float(str(row[amount_col]).replace(",", "") or 0)
        except:
            continue

        # 日付のズレが5日以内かチェック
        delta_days = abs((d_obj - ocr_date).days)
        if delta_days > 5 or amt != ocr_amt:
            # 5日以上ずれる or 金額不一致なら次の行へ
            continue

        # CSVの店舗名正規化
        raw_csv = str(row[store_col] or "")
        # ① 英字BRAND_MAP優先
        # (A) 英字正規化→BRAND_MAP（空白を残してキーを合わせる）
        eng = normalize_text(raw_csv).upper()
        eng = re.sub(r'(?:-SS|/NFC)$', '', eng)  # 接尾辞だけ削除
        if eng in BRAND_MAP:
            csv_kana = BRAND_MAP[eng]
        else:
            # ② 接頭辞一致（OCRが短い場合）
            kana_tmp_csv = to_katakana(raw_csv)
            kana_tmp_csv = canonical_kana(kana_tmp_csv)
            key_csv = normalize_text(kana_tmp_csv).upper().replace(" ", "")
            key_csv = re.sub(r'(?:-SS|/NFC)$', '', key_csv)
            if kana_tmp_csv.startswith(ocr_kana) and len(ocr_kana) >= 3:
                csv_kana = kana_tmp_csv
            else:
                # ③ カタカナ→BRAND_MAP
                csv_kana = BRAND_MAP.get(key_csv, kana_tmp_csv)

        # デバッグ出力
        print(f"[DEBUG] 行{idx} → OCR:'{ocr_kana}', CSV:'{csv_kana}'")

        # 厳密一致 or 部分一致
        if ocr_kana == csv_kana:
            df.at[idx, "レシートチェック"] = True
            matched = True
            break
        else:
            # RapidFuzz で部分一致
            scores = {
                "partial_ratio": fuzz.partial_ratio(ocr_kana, csv_kana),
                "token_sort":    fuzz.token_sort_ratio(ocr_kana, csv_kana),
                "token_set":     fuzz.token_set_ratio(ocr_kana, csv_kana)
            }
            method = max(scores, key=scores.get)
            score  = scores[method]
            print(f"[DEBUG]    {method}={score:.1f}")
            if score >= FUZZ_THRESHOLD:
                df.at[idx, "レシートチェック"] = True
                matched = True
                break
            else:
                mismatches.append({
                    "csv":       csv_path.name,
                    "row":       idx,
                    "ocr_store": ocr_kana,
                    "csv_store": csv_kana,
                    "method":    method,
                    "score":     score
                })

    # 上書き保存
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return matched


def main():
    """
    メイン処理:
      1. レシート画像をOCR＆ファジーマッチング
      2. CSVを更新し、一致・不一致パターンをそれぞれファイルに保存
      3. 一致した画像は matched フォルダへ移動
    """
    RECEIPT_FOLDER.mkdir(parents=True, exist_ok=True)
    matched_dir = RECEIPT_FOLDER / "matched"
    matched_dir.mkdir(exist_ok=True)

    all_matched = []
    all_unmatched = []

    for img_path in RECEIPT_FOLDER.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        print(f"\n=== 画像処理: {img_path.name} ===")
        try:
            info = extract_receipt_info(img_path)
            # store_kana（BRAND_MAP でマップ済み）を表示する
            print(f"抽出結果 → 日付:{info['date_obj']} 店舗:{info['store_kana']} 金額:{info['amount']:,}円")
        except Exception as e:
            print(f"[ERROR] OCR失敗: {e}")
            continue

        matched = False
        for csv_file in CSV_FOLDER.glob("*.csv"):
            # mismatches 引数を all_unmatched に渡す
            ok = mark_receipt_in_csv(csv_file, info, all_unmatched)
            print(f"→ CSV処理: {csv_file.name} → {'✅' if ok else '❌'}")
            if ok:
                matched = True
                all_matched.append({
                    "image": img_path.name,
                    "csv": csv_file.name,
                    "date": info["date_obj"].isoformat(),
                    "store": info["store_kana"],
                    "amount": info["amount"]
                })
                shutil.move(str(img_path), matched_dir / img_path.name)
                break

        if not matched:
            # どのCSVにもマッチしなかったレシート自体を記録
            all_unmatched.append({
                "image": img_path.name,
                "date": info["date_obj"].isoformat(),
                "store": info["store_kana"],
                "amount": info["amount"]
            })

    # マッチパターンの一括保存
    if all_matched:
        out_matched = RECEIPT_FOLDER / "matched_patterns.csv"
        pd.DataFrame(all_matched).to_csv(out_matched, index=False, encoding="utf-8-sig")
        print(f"[INFO] マッチパターンを保存: {out_matched}")

    # 不一致パターンの一括保存
    if all_unmatched or all_unmatched:
        # all_unmatched にはレシート未マッチ情報と、
        # mark_receipt_in_csv 内での行レベルの不一致も混在しています
        out_unmatched = RECEIPT_FOLDER / "unmatched_patterns.csv"
        pd.DataFrame(all_unmatched).to_csv(out_unmatched, index=False, encoding="utf-8-sig")
        print(f"[INFO] 不一致パターンを保存: {out_unmatched}")


if __name__ == "__main__":
    main()
