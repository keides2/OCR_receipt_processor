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
from pykakasi import kakasi
import unicodedata
import openai
import os

# 全ての DeprecationWarning を無視
warnings.simplefilter("ignore", DeprecationWarning)

# OpenAI API設定
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 環境変数が設定されていません")
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1",  # 明示的にベースURLを指定
    timeout=60.0  # タイムアウトを設定
)

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
vision_client = vision.ImageAnnotatorClient()
RECEIPT_FOLDER = Path(r"G:\マイドライブ\receipt\receipt_img")
CSV_FOLDER = Path(r"G:\マイドライブ\receipt\csv_statements")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# キーは正規化＋大文字化した raw_store、値は正しいカタカナ
BRAND_MAP = {
    "MOS BURGER": "モスノネツトチユウモン",
    "KFC": "ケンタツキ-フライドチキン",
    "オンモールサカイテッポウマチ": "イオンモールサカイテッポウチヨウ",
    "コーナン": "コーナン商事",
    "MINIサカイ": "アルコンサカイ",
    "Toys us": "トイザらス",

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


def get_similarity_score_with_chatgpt(ocr_kana, csv_kana):
    """
    ChatGPTを使用して文字列の類似度を計算します。

    Args:
        ocr_kana (str): OCRで抽出された店舗名
        csv_kana (str): CSVに記載された店舗名

    Returns:
        float: 類似度スコア（0〜100）
    """
    prompt = (
        f"以下の2つの日本語文字列の類似度を0から100のスコアで評価してください。\n"
        f"1. {ocr_kana}\n"
        f"2. {csv_kana}\n"
        f"スコアのみを返してください。"
    )

    try:
        # クォータ超過エラー時は文字列の単純比較にフォールバック
        if ocr_kana == csv_kana:
            return 100
        elif normalize_spaces(ocr_kana) == normalize_spaces(csv_kana):
            return 95
        # 部分一致チェック
        ocr_norm = normalize_spaces(ocr_kana)
        csv_norm = normalize_spaces(csv_kana)
        if ocr_norm in csv_norm or csv_norm in ocr_norm:
            return 85

        # text-embedding-3-small で埋め込みベクトルを取得しコサイン類似度を計算
        emb_ocr = client.embeddings.create(
            model="text-embedding-3-small",
            input=ocr_kana
        ).data[0].embedding
        emb_csv = client.embeddings.create(
            model="text-embedding-3-small",
            input=csv_kana
        ).data[0].embedding

        # コサイン類似度計算
        import numpy as np
        v1 = np.array(emb_ocr)
        v2 = np.array(emb_csv)
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        # [-1,1] → [0,100]スコア化
        score = int((cos_sim + 1) * 50)
        return score
    except Exception as e:
        if "insufficient_quota" in str(e):
            print("[WARNING] APIクォータ超過。文字列の単純比較にフォールバック")
            # 文字列の長さに基づく簡易スコア計算
            len_ratio = min(len(ocr_kana), len(csv_kana)) / max(len(ocr_kana), len(csv_kana))
            return int(len_ratio * 70)  # 最大70点
        else:
            print(f"[ERROR] ChatGPT API呼び出し失敗: {e}")
            return 0


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
    for pat in [
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",  # 西暦形式
        r"(\d{4})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})",  # 西暦スラッシュ形式
        r"(\d{2})年(\d{1,2})月(\d{1,2})日"  # 元号形式（例: 25年01月18日）
    ]:
        m = re.search(pat, s)
        if m:
            y, M, d = m.groups()
            if len(y) == 2:  # 元号形式の場合
                y = str(2000 + int(y))  # 2000年以降と仮定
            return datetime.date(int(y), int(M), int(d))
    return None


def preprocess_image(path: Path) -> bytes:
    img = Image.open(path).convert("L").filter(ImageFilter.SHARPEN)
    bw  = img.point(lambda x: 0 if x < 140 else 255, '1')
    buf = BytesIO(); bw.save(buf, format="PNG")
    return buf.getvalue()


def normalize_store_name_with_gpt(raw_store: str) -> str:
    """ChatGPTを使用して店舗名を正規化します"""
    try:
        prompt = (
            f"以下の店舗名を正規化してカタカナで返してください。固有名詞は維持してください：\n"
            f"{raw_store}\n"
            f"カタカナのみを返してください。"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは店舗名を正規化する日本語アシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0
        )
        normalized = response.choices[0].message.content.strip()
        print(f"[DEBUG] GPT正規化: {raw_store} → {normalized}")
        return normalized
    except Exception as e:
        print(f"[WARNING] 店舗名正規化スキップ: {e}")
        return raw_store


def clean_store_name(raw_name: str) -> str:
    """店舗名から不要な文字を除去して正規化"""
    # カード売上などの不要な文字を除去
    cleaned = re.sub(r'(カード)?売上.*$', '', raw_name)
    # 記号除去
    cleaned = re.sub(r'[【】\[\]()（）「」『』《》〈〉｛｝\{\}]', '', raw_name)
    # 末尾の記号を除去
    cleaned = re.sub(r'[】\]）>〉》\}｝]+$', '', cleaned)
    return cleaned.strip()

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

    # デバッグ: OCR全文・行リスト出力
    print("----- OCR全文 -----")
    print(text)
    print("----- OCR行リスト -----")
    for i, ln in enumerate(lines):
        print(f"[OCR line {i}]: {ln}")

    # 「加盟店名」や「ミズラボ」でフィルタ
    found_kameiten = any("加盟店名" in ln for ln in lines)
    found_mizulabo = any("ミズラボ" in ln for ln in lines)
    print(f"[DEBUG] 加盟店名含む行あり: {found_kameiten}")
    print(f"[DEBUG] ミズラボ含む行あり: {found_mizulabo}")

    # --- 1) 日付抽出（デバッグ付き） ---
    date_obj = None

    # --- 1a-1) 「ご利用日」ラベル以降の全行を順にチェックして日付を探す ---
    for idx, ln in enumerate(lines):
        if "ご利用日" in ln:
            print(f"[DEBUG] ご利用日ラベル行: line[{idx}]={ln!r}")
            # ラベル以降10行をスキャン
            for j, nxt in enumerate(lines[idx+1:idx+11], start=idx+1):
                print(f"[DEBUG]  ラベル後行 line[{j}]={nxt!r}")
                # 年月日＋時刻も許容
                m = re.match(r"([0-9]{2,4})年(\d{1,2})月(\d{1,2})日(?:\s*\d{1,2}:\d{2})?", nxt)
                if not m:
                    m = re.match(r"([0-9]{2,4})[\/\-\.(](\d{1,2})[\/\-\.)](\d{1,2})(?:\s*\d{1,2}:\d{2})?", nxt)
                if not m:
                    continue
                y_t, M_t, d_t = m.groups()[:3]
                year  = int(y_t) if len(y_t)==4 else 2000 + int(y_t)
                month = int(M_t); day = int(d_t)
                print(f"[DEBUG]  ラベル後行マッチ groups={m.groups()} → {year}-{month}-{day}")
                if 1 <= month <= 12 and 1 <= day <= 31:
                    date_obj = datetime.date(year, month, day)
                    break
            if date_obj:
                break

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
    # 「加盟店名」行がある場合はその直後の行を優先
    raw_store = ""
    for idx, ln in enumerate(lines):
        if "加盟店名" in ln:
            # 「加盟店名」の直後の行が店舗名であることが多い
            if idx + 1 < len(lines):
                candidate = lines[idx + 1].strip()
                # 5文字以上のカタカナ or カタカナ＋漢字混在 or 「ミズラボ」など特定ワード
                if (re.fullmatch(r"[ァ-ヴー]{5,}", candidate)
                    or (re.search(r"[一-龯]", candidate) and re.search(r"[ァ-ヴー]", candidate))
                    or "ミズラボ" in candidate):
                    raw_store = candidate
                    print(f"[DEBUG] 加盟店名直後候補: {raw_store}")
                    break
            # 「加盟店名」行自体に店舗名が含まれる場合も考慮
            m = re.search(r'加盟店名[：:\s]*([^\d\s]+)', ln)
            if m:
                raw_store = m.group(1).strip()
                print(f"[DEBUG] 加盟店名行内候補: {raw_store}")
                break

    # fallback: 既存の推定ロジック
    if not raw_store:
        candidates = []
        for ln in lines:
            if (re.search(r"[一-龯]", ln) and re.search(r"[ァ-ヴー]", ln)) or re.fullmatch(r"[ァ-ヴー]{5,}", ln):
                candidates.append(ln)
        if not candidates:
            for ln in lines:
                if re.fullmatch(r"[ァ-ヴー]+", ln):
                    candidates.append(ln)
        if not candidates:
            for ln in lines:
                if re.fullmatch(r"[A-Za-z\s]+", ln):
                    candidates.append(ln)
        if candidates:
            raw_store = candidates[0].strip()
        elif lines:
            raw_store = lines[0].strip()

    print(f"[DEBUG] 店舗名候補: {raw_store}")

    # クリーニング処理
    raw_store = clean_store_name(raw_store)

    # --- 4) 店舗名マッピング ---
    eng = normalize_spaces(raw_store).upper()
    eng = re.sub(r'(?:-SS|/NFC)$', '', eng)
    if eng in BRAND_MAP:
        store_kana = BRAND_MAP[eng]
    else:
        try:
            # GPTで正規化を試みる
            normalized = normalize_store_name_with_gpt(raw_store)
            kana_tmp = canonical_kana(normalized)
        except Exception as e:
            print(f"[WARNING] GPT正規化スキップ: {e}")
            # フォールバック：通常のカタカナ変換
            kana_tmp = canonical_kana(to_katakana(raw_store))

        key = normalize_spaces(kana_tmp).upper()
        key = re.sub(r'(?:-SS|/NFC)$', '', key)
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

    Args:
        csv_path (Path): CSVファイルパス
        info (dict): extract_receipt_info の戻り値
        mismatches (list): 不一致パターン記録用リスト

    Returns:
        bool: 一致した行があれば True、なければ False
    """
    import pandas as pd

    # OCR結果
    ocr_date = info["date_obj"]
    ocr_amt  = info["amount"]
    ocr_kana = info["store_kana"]

    # CSV 読み込み
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp932", dtype=str)

    # 列検出
    date_col   = next((c for c in df.columns if c in ["利用日","取引日","取引年月日"]), None)
    store_col  = next((c for c in df.columns if c in ["利用店名","利用店名・商品名","店舗名","加盟店名"]), None)
    amount_col = next((c for c in df.columns if c in ["利用金額","金額","取引金額"]), None)

    # 必須列が見つからない場合はスキップ
    if date_col is None or store_col is None or amount_col is None:
        print(f"[WARNING] 必須列が見つかりません: date_col={date_col}, store_col={store_col}, amount_col={amount_col} in {csv_path.name}")
        return False

    # チェック列追加
    if "レシートチェック" not in df.columns:
        df["レシートチェック"] = False

    matched = False
    for idx, row in df.iterrows():
        # 日付・金額の照合（±5日以内を許容）
        d_obj = parse_date(str(row[date_col]))
        if d_obj is None:
            continue
        try:
            amt = float(str(row[amount_col]).replace(",", "") or 0)
        except:
            continue

        delta_days = abs((d_obj - ocr_date).days)
        if delta_days > 5 or amt != ocr_amt:
            continue

        # CSVの店舗名正規化
        raw_csv = str(row[store_col] or "")
        csv_kana = BRAND_MAP.get(raw_csv.upper(), raw_csv)

        # ChatGPTで類似度スコアを計算
        score = get_similarity_score_with_chatgpt(ocr_kana, csv_kana)
        print(f"[DEBUG] 行{idx} → OCR:'{ocr_kana}', CSV:'{csv_kana}', Score:{score}")

        if score >= 30:  # スコアが30以上で一致とみなす
            df.at[idx, "レシートチェック"] = True
            matched = True
            break
        else:
            mismatches.append({
                "csv":       csv_path.name,
                "row":       idx,
                "ocr_store": ocr_kana,
                "csv_store": csv_kana,
                "score":     score
            })

    # 上書き保存
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return matched


def main():
    """
    メイン処理:
      1. レシート画像をOCR＆ChatGPTマッチング
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
