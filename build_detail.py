"""
FEXTA 상세페이지 자동화 공장 (GitHub Actions 전용 4-에이전트 파이프라인)
- 에이전트 1: 카페24 상품 URL 원본 스펙·고화질 이미지·호환 부자재·공장 8대 공정 공용 URL 수집
- 에이전트 2: 서버 전용 딥러닝 AI(ISNet) 정밀 누끼 + [소/중/대] 뱃지 제거 + 1200~1600px 고화질 선명화
- 에이전트 3: SEO · AEO · GEO 최적화 B2B 카피, 4대 Q&A 및 Product+FAQPage JSON-LD 생성
- 에이전트 4: 수석 디자이너 100% 인라인 카페24 HTML + 로컬 검수 미리보기 + WebFTP 업로드 가이드 출력
"""

import io
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote, urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from rembg import new_session, remove
from scipy import ndimage

# =============================================================================
# 기본 설정 및 표준 상수
# =============================================================================
ALLOWED_MATERIALS = ["마닐라지", "합지", "크라프트지", "수입지"]

FACTORY_PROCESSES = [
    {"num": "01", "title": "샘플커팅기", "desc": "생산 전 샘플커팅기로 제품을 확인합니다.", "file": "factory_01.jpg"},
    {"num": "02", "title": "옵셋인쇄기", "desc": "원단을 CTP로 장착하여 도수대로 인쇄합니다.", "file": "factory_02.jpg"},
    {"num": "03", "title": "코팅기", "desc": "인쇄된 원단을 코팅합니다.", "file": "factory_03.jpg"},
    {"num": "04", "title": "합지기", "desc": "인쇄지와 골판지를 접착제를 사용해 맞붙입니다.", "file": "factory_04.jpg"},
    {"num": "05", "title": "톰슨기", "desc": "목형을 이용해 원단을 원하는 모양으로 커팅합니다.", "file": "factory_05.jpg"},
    {"num": "06", "title": "자동접착기", "desc": "톰슨된 원단을 레일을 타고 접고 붙입니다.", "file": "factory_06.jpg"},
    {"num": "07", "title": "제함기", "desc": "치구를 이용해 모양을 만들면서 접착합니다.", "file": "factory_07.jpg"},
    {"num": "08", "title": "물류", "desc": "완성된 제품을 보관하고 출고합니다.", "file": "factory_08.jpg"},
]

DEFAULT_CAPTIONS = [
    "정면 외관 및 선명한 패키지 인쇄 디테일",
    "측면 구조 · 원산지 표기란 및 숨구멍 마감",
    "내부 수납 공간 및 원단 마감 확인",
    "간편 조립 및 매장 출고 패키지 형태",
    "실물 디테일 클로즈업 컷",
    "매장 배달 포장 연출 컷",
]

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Referer": "https://fexta.co.kr/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HTTP_HEADERS)


def normalize_url(url: str) -> str:
    """상대 경로 또는 프로토콜 생략 URL을 https://fexta.co.kr 절대 URL로 정규화"""
    if not url:
        return ""
    u = url.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return "https://fexta.co.kr" + u
    if not u.startswith("http"):
        return urljoin("https://fexta.co.kr/", u)
    return u


def fetch_html(url: str, timeout: int = 20) -> str:
    """fexta.co.kr HTML을 인코딩 깨짐 없이 수집"""
    resp = SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def download_image(url: str, timeout: int = 20) -> Image.Image | None:
    """원본 이미지를 메모리(PIL Image)로 다운로드"""
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        return img
    except Exception as exc:
        print(f"[WARN] 이미지 다운로드 실패 ({url}): {exc}")
        return None


# =============================================================================
# 에이전트 1: 카페24 상품 스펙 · 고화질 이미지 · 호환 부자재 · 공장 공용 URL 수집기
# =============================================================================
def recommend_sub_materials(name: str, w_str: str, material: str, addon_text: str) -> dict:
    """상품명·규격·추가구매상품 기반으로 호환 트레이 및 비닐봉투 매칭"""
    compact = re.sub(r"\s+", "", name or "")
    w = int(w_str) if str(w_str).isdigit() else 0

    detected_whole = ""
    if addon_text:
        m = re.search(r"트레이\s*\d+호(?:\s*\([^)]+\))?", addon_text)
        if m:
            detected_whole = m.group(0).strip()

    if "블랙쉐프" in compact:
        if "소" in compact:
            return {"whole": detected_whole or "트레이 2호 (172×119×55mm)", "half": "해당없음", "bag": "링펀치 400 · 시스루 중"}
        if "중" in compact:
            return {"whole": detected_whole or "트레이 16호 (215×161×50mm)", "half": "트레이 17호 (163×107×50mm)", "bag": "링펀치 대 · 시스루 대"}
        return {"whole": detected_whole or "트레이 6호 (240×140×55mm)", "half": "트레이 2호 (172×119×55mm)", "bag": "시스루 대 · 특대 봉투"}

    if "아일라킷" in compact:
        if "소" in compact:
            return {"whole": detected_whole or "트레이 7호", "half": "해당없음", "bag": "링펀치 400 · 아일라킷 소"}
        if "중" in compact:
            return {"whole": detected_whole or "트레이 2호", "half": "해당없음", "bag": "링펀치 400 · 시스루 중"}
        if "특대" in compact:
            return {"whole": detected_whole or "트레이 6호", "half": "트레이 7호 · 트레이 8호", "bag": "시스루 대 · 아일라킷 대"}
        return {"whole": detected_whole or "트레이 3호", "half": "트레이 9호", "bag": "시스루 대 · 아일라킷 대"}

    if "금계" in compact:
        if "소" in compact:
            return {"whole": detected_whole or "트레이 8호", "half": "해당없음", "bag": "링펀치 400 · 금계 일반"}
        if "중" in compact:
            return {"whole": detected_whole or "트레이 2호", "half": "해당없음", "bag": "링펀치 400 · 시스루 중"}
        if "특대" in compact:
            return {"whole": detected_whole or "트레이 16호", "half": "트레이 7호 · 트레이 8호", "bag": "시스루 대 · 금계 특대"}
        return {"whole": detected_whole or "트레이 3호", "half": "트레이 9호", "bag": "링펀치 400 · 시스루 대 · 금계 일반"}

    if "옐로우" in compact or material == "합지" or w >= 225:
        return {"whole": detected_whole or "트레이 16호", "half": "트레이 7호 · 트레이 8호", "bag": "시스루 대 · 특대 봉투"}
    if 0 < w <= 185:
        return {"whole": detected_whole or "트레이 2호", "half": "해당없음", "bag": "링펀치 400 · 시스루 중"}
    if 185 < w <= 212:
        return {"whole": detected_whole or "트레이 3호", "half": "트레이 9호", "bag": "링펀치 400 · 시스루 대"}
    return {"whole": detected_whole or "트레이 16호", "half": "트레이 7호 · 트레이 8호", "bag": "시스루 대"}


def search_mall_thumbnail(keyword: str) -> str:
    """자사몰 검색 결과에서 트레이/비닐봉투 기존 썸네일 URL을 직접 가져와 FTP 용량 0MB 재사용"""
    clean_kw = re.sub(r"\(.*?\)", "", keyword or "").split("·")[0].strip()
    if not clean_kw or "해당없음" in clean_kw:
        return ""
    search_url = f"https://fexta.co.kr/product/search.html?keyword={quote(clean_kw)}"
    try:
        html = fetch_html(search_url, timeout=15)
        soup = BeautifulSoup(html, "lxml")
        for selector in [
            ".prdList img[src*='/web/product/']",
            ".thumbnail img[src*='/web/product/']",
            "img[src*='/web/product/medium/']",
            "img[src*='/web/product/big/']",
        ]:
            img = soup.select_one(selector)
            if img:
                src = img.get("ec-data-src") or img.get("data-original") or img.get("src")
                if src:
                    return normalize_url(src)
    except Exception as exc:
        print(f"[WARN] 부자재 썸네일 검색 스킵 ({clean_kw}): {exc}")
    return ""


def fetch_shared_factory_urls() -> list[str]:
    """기존 블랙쉐프(1463)·금계(3178) 상세페이지에 올라간 공장 8대 공정 공용 URL 자동 재사용"""
    reference_pages = [
        "https://fexta.co.kr/product/detail.html?product_no=1463",
        "https://fexta.co.kr/product/detail.html?product_no=3178",
    ]
    for url in reference_pages:
        try:
            html = fetch_html(url, timeout=15)
            soup = BeautifulSoup(html, "lxml")
            detail = soup.select_one("#prdDetail")
            if not detail:
                continue
            imgs = []
            for img in detail.select("img"):
                src = img.get("ec-data-src") or img.get("data-src") or img.get("src")
                if src:
                    norm = normalize_url(src)
                    if norm not in imgs and not re.search(r"icon_|btn_|logo|naver|kakao", norm, re.I):
                        imgs.append(norm)
            named = [u for u in imgs if re.search(r"factory|process|cut|offset|coat|lamin|thomson|glue|box|logi", u, re.I)]
            if len(named) >= 8:
                return named[:8]
            if len(imgs) >= 8:
                return imgs[-8:]
        except Exception as exc:
            print(f"[WARN] 공장 공용 URL 탐색 스킵 ({url}): {exc}")

    # 폴백: 표준 공용 WebFTP 경로
    return [f"/web/upload/detail/common/factory/{p['file']}" for p in FACTORY_PROCESSES]


def scrape_cafe24_product(product_url: str, material_override: str = "자동 판별") -> dict:
    """에이전트 1 메인: 카페24 상품 정보 및 모든 후보 이미지 수집"""
    print(f"[AGENT 1] 상품 페이지 접속 중: {product_url}")
    html = fetch_html(product_url)
    soup = BeautifulSoup(html, "lxml")

    no_match = re.search(r"product_no=(\d+)", product_url) or re.search(r"/(\d+)/?$", product_url)
    product_no = no_match.group(1) if no_match else "custom"

    # 상품명 추출
    h2 = soup.select_one(".infoArea h2, .headingArea h2")
    og_title = soup.select_one('meta[property="og:title"]')
    raw_title = (
        (h2.get_text(strip=True) if h2 else "")
        or (og_title.get("content", "").strip() if og_title else "")
        or (soup.title.get_text(strip=True) if soup.title else "")
    )
    clean_name = re.sub(r"\s*-\s*팩스타.*$", "", raw_title, flags=re.I)
    clean_name = clean_name.replace("(해외배송 가능상품)", "").strip() or f"팩스타 상품 (No.{product_no})"

    # 기본 정보 테이블 파싱
    spec_text, pack_qty, unit_price, sale_price = "", "", "", 0
    for tr in soup.select("table tr"):
        th = tr.select_one("th")
        td = tr.select_one("td")
        if not th or not td:
            continue
        k = th.get_text(" ", strip=True)
        v = td.get_text(" ", strip=True)
        if not k or not v:
            continue
        if re.search(r"규격|사이즈|크기", k) and not spec_text:
            spec_text = v
        if re.search(r"입수|수량|단위", k) and not pack_qty and re.search(r"\d", v):
            pack_qty = v
        if re.search(r"개당가|개당", k) and not unit_price:
            unit_price = v
        if "판매가" in k and sale_price == 0:
            m = re.search(r"(\d{3,7})", v.replace(",", ""))
            if m:
                sale_price = int(m.group(1))

    # 추가구매상품(트레이) 텍스트 파싱
    addon_el = soup.select_one(".xans-product-addproduct, #prdAddProduct")
    if addon_el:
        addon_text = addon_el.get_text(" ", strip=True)
    else:
        addon_text = " ".join(
            opt.get_text(strip=True) for opt in soup.select("select option")
            if re.search(r"트레이|봉투", opt.get_text(strip=True))
        )

    prd_detail = soup.select_one("#prdDetail")
    detail_head_text = prd_detail.get_text(" ", strip=True)[:1500] if prd_detail else ""
    dim_match = re.search(r"(\d{2,3})\s*[xX×*]\s*(\d{2,3})\s*[xX×*]\s*(\d{2,3})", f"{spec_text} {clean_name} {detail_head_text}")
    w_val = dim_match.group(1) if dim_match else "-"
    d_val = dim_match.group(2) if dim_match else "-"
    h_val = dim_match.group(3) if dim_match else "-"

    # 원단 4종 표준 판별 (마닐라지 / 합지 / 크라프트지 / 수입지)
    if material_override in ALLOWED_MATERIALS:
        material = material_override
    else:
        mat_ctx = f"{clean_name} {spec_text} {detail_head_text[:600]}"
        if "크라프트" in mat_ctx:
            material = "크라프트지"
        elif "수입지" in mat_ctx:
            material = "수입지"
        elif re.search(r"합지|골판지|옐로우|블랙쉐프", mat_ctx) and "마닐라" not in clean_name:
            material = "합지"
        else:
            material = "마닐라지"

    # 입수량 / 개당가 조합
    qty_m = re.search(r"(\d{2,4})\s*(?:매|개입|개)", pack_qty or clean_name)
    qty_num = int(qty_m.group(1)) if qty_m else 0
    if not pack_qty and qty_num > 0:
        pack_qty = f"1박스 {qty_num}매입"
    if not unit_price and sale_price > 0 and qty_num > 0:
        unit_price = f"약 {round(sale_price / qty_num):,}원"
    qty_display = (
        f"{pack_qty} (개당 {re.sub(r'^개당\\s*', '', unit_price)})"
        if pack_qty and unit_price
        else (pack_qty or "1박스 묶음")
    )

    # 대표 이미지 & 추가 이미지 & 본문 이미지 수집
    main_el = soup.select_one('.keyImg img, .thumbnail img, img[src*="/web/product/big/"]')
    og_img = soup.select_one('meta[property="og:image"]')
    main_img_url = normalize_url(
        (main_el.get("src") if main_el else "")
        or (og_img.get("content") if og_img else "")
    )

    seen = set()
    if main_img_url:
        seen.add(main_img_url)

    detail_img_urls = []
    # 1) 추가 썸네일(.listImg)을 고화질 big 경로로 업그레이드
    for img in soup.select(".listImg img"):
        raw_src = img.get("ec-data-src") or img.get("data-original") or img.get("src")
        if not raw_src or re.search(r"icon_|btn_|arrow|blank", raw_src, re.I):
            continue
        upgraded = normalize_url(raw_src).replace("/web/product/small/", "/web/product/big/").replace("/web/product/tiny/", "/web/product/big/")
        if upgraded not in seen:
            seen.add(upgraded)
            detail_img_urls.append(upgraded)

    # 2) 본문 상세 이미지(#prdDetail img)
    if prd_detail:
        for img in prd_detail.select("img"):
            raw_src = img.get("ec-data-src") or img.get("data-original") or img.get("data-src") or img.get("src")
            alt = img.get("alt") or ""
            if not raw_src:
                continue
            norm = normalize_url(raw_src)
            if re.search(r"icon_|btn_|logo|naver|kakao|arrow|bullet|blank|spacer|talk|plus_id|factory|샘플커팅|옵셋인쇄|코팅기|합지기|톰슨기|자동접착|제함기|물류", f"{norm} {alt}", re.I):
                continue
            if norm not in seen:
                seen.add(norm)
                detail_img_urls.append(norm)

    # 호환 부자재 및 자사몰 썸네일 자동 수집
    sub = recommend_sub_materials(clean_name, w_val, material, addon_text)
    sub_thumbs = {
        "whole": search_mall_thumbnail(sub["whole"]),
        "half": search_mall_thumbnail(sub["half"]),
        "bag": search_mall_thumbnail(sub["bag"]),
    }

    factory_urls = fetch_shared_factory_urls()

    return {
        "product_no": product_no,
        "product_url": product_url,
        "name": clean_name,
        "w": w_val,
        "d": d_val,
        "h": h_val,
        "dim_text": f"{w_val} × {d_val} × {h_val} mm" if w_val != "-" else "상세 규격 문의",
        "material": material,
        "quantity": qty_display,
        "usage": "치킨, 닭강정, 순살, 외식 배달 및 테이크아웃 포장 전용",
        "sub": sub,
        "sub_thumbs": sub_thumbs,
        "main_img_url": main_img_url,
        "detail_img_urls": detail_img_urls,
        "factory_urls": factory_urls,
    }


# =============================================================================
# 에이전트 2: 서버 전용 AI 딥러닝 정밀 누끼(ISNet) + 뱃지 제거 + 고화질 선명화
# =============================================================================
def init_rembg_session():
    """직선 패키지 모서리 분리에 강한 ISNet 신경망 모델 우선 로드"""
    try:
        print("[AGENT 2] AI 신경망 누끼 모델(isnet-general-use) 초기화 중...")
        return new_session("isnet-general-use")
    except Exception as exc:
        print(f"[WARN] ISNet 로드 실패, 기본 u2net으로 폴백: {exc}")
        return new_session("u2net")


def remove_detached_badges_and_defringe(rgba_img: Image.Image) -> Image.Image:
    """
    1) 연결 요소 분석(Connected Components)으로 우측 상단 [소/중/대] 원형 뱃지 및 잡티 완벽 삭제
    2) 어두운 패키지 테두리에 남는 흰색 경계 띠(Halo) 디프린지 처리
    """
    arr = np.array(rgba_img.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]

    # 미세 알파 노이즈 정리
    alpha[alpha < 20] = 0
    alpha[alpha > 242] = 255

    # 연결 요소 분석: 가장 큰 본체(박스) 면적의 14% 미만인 동떨어진 조각([소/중/대] 뱃지) 삭제
    binary = alpha > 35
    labeled, num_features = ndimage.label(binary)
    if num_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        max_size = float(np.max(sizes)) if len(sizes) > 0 else 0.0
        for idx, sz in enumerate(sizes, start=1):
            if sz < max_size * 0.14:
                alpha[labeled == idx] = 0

    arr[:, :, 3] = alpha

    # 경계 디프린지 (반투명 픽셀이 배경색 흰 띠를 머금지 않도록 내부 불투명 색상을 약간 당겨옴)
    edge_mask = (alpha > 0) & (alpha < 225)
    if np.any(edge_mask):
        # 내부 픽셀 기준 최대 필터로 RGB 보간
        solid = alpha >= 225
        for c in range(3):
            channel = arr[:, :, c]
            blurred = ndimage.uniform_filter(np.where(solid, channel, 0).astype(np.float32), size=3)
            weight = ndimage.uniform_filter(solid.astype(np.float32), size=3)
            safe_fill = np.where(weight > 0.05, blurred / np.maximum(weight, 1e-5), channel)
            arr[:, :, c] = np.where(edge_mask, np.clip(channel * 0.3 + safe_fill * 0.7, 0, 255), channel).astype(np.uint8)

    return Image.fromarray(arr, mode="RGBA")


def create_studio_highres_cutout(
    src_img: Image.Image,
    rembg_session,
    out_size: int = 1200,
    bg_mode: str = "studio",
) -> Image.Image:
    """
    원본 이미지 → AI 신경망 누끼 → [소/중/대] 뱃지 제거 → 정중앙 1200~1600px 배치
    → Unsharp Mask 고화질 선명화 → 스튜디오 배경 + 이중 바닥 그림자 합성
    """
    # 1. 신경망 AI 배경 제거
    cutout_rgba = remove(src_img.convert("RGB"), session=rembg_session, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=15, alpha_matting_erode_size=6)
    cutout_rgba = remove_detached_badges_and_defringe(cutout_rgba)

    # 2. 순수 제품 본체 바운딩 박스 정밀 계산
    bbox = cutout_rgba.getbbox()
    if bbox:
        cropped = cutout_rgba.crop(bbox)
    else:
        cropped = cutout_rgba

    # 3. 정중앙 배치 (LANCZOS 최고급 리샘플링)
    pad = int(out_size * 0.085)
    max_inner = out_size - pad * 2
    scale = min(max_inner / cropped.width, max_inner / cropped.height)
    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 4. 고화질 선명화 (Unsharp Mask + 미세 대비 보정 — 박스 인쇄 글씨 복원)
    r_ch, g_ch, b_ch, a_ch = resized.split()
    rgb_only = Image.merge("RGB", (r_ch, g_ch, b_ch))
    rgb_only = rgb_only.filter(ImageFilter.UnsharpMask(radius=1.6, percent=145, threshold=2))
    rgb_only = ImageEnhance.Contrast(rgb_only).enhance(1.06)
    rgb_only = ImageEnhance.Color(rgb_only).enhance(1.04)
    sharpened = Image.merge("RGBA", (*rgb_only.split(), a_ch))

    draw_x = (out_size - new_w) // 2
    draw_y = (out_size - new_h) // 2

    if bg_mode == "transparent":
        canvas = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
        canvas.paste(sharpened, (draw_x, draw_y), sharpened)
        return canvas

    # 5. 스튜디오 배경 생성 (방사형 라이트 그라데이션 or 순백색)
    if bg_mode == "white":
        bg = Image.new("RGBA", (out_size, out_size), (255, 255, 255, 255))
    else:
        yy, xx = np.ogrid[:out_size, :out_size]
        dist = np.sqrt(((xx - out_size * 0.5) / (out_size * 0.72)) ** 2 + ((yy - out_size * 0.45) / (out_size * 0.72)) ** 2)
        t = np.clip(dist, 0.0, 1.0)
        r_bg = (255 * (1 - t) + 242 * t).astype(np.uint8)
        g_bg = (255 * (1 - t) + 240 * t).astype(np.uint8)
        b_bg = (255 * (1 - t) + 234 * t).astype(np.uint8)
        a_bg = np.full((out_size, out_size), 255, dtype=np.uint8)
        bg = Image.fromarray(np.stack([r_bg, g_bg, b_bg, a_bg], axis=-1), mode="RGBA")

    # 6. 현실적인 이중 바닥 그림자 (Ambient + Contact Occlusion)
    shadow_layer = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_layer)
    cx = out_size // 2
    bottom_y = draw_y + new_h

    # 넓은 확산 그림자
    rx1, ry1 = int(new_w * 0.42), int(out_size * 0.022)
    s_draw.ellipse([cx - rx1, bottom_y - ry1 - int(out_size * 0.008), cx + rx1, bottom_y + ry1 - int(out_size * 0.008)], fill=(18, 18, 17, 34))
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=int(out_size * 0.02)))

    # 좁은 접지 그림자
    contact_layer = Image.new("RGBA", (out_size, out_size), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(contact_layer)
    rx2, ry2 = int(new_w * 0.34), int(out_size * 0.011)
    c_draw.ellipse([cx - rx2, bottom_y - ry2 - int(out_size * 0.004), cx + rx2, bottom_y + ry2 - int(out_size * 0.004)], fill=(18, 18, 17, 46))
    contact_layer = contact_layer.filter(ImageFilter.GaussianBlur(radius=int(out_size * 0.008)))

    bg = Image.alpha_composite(bg, shadow_layer)
    bg = Image.alpha_composite(bg, contact_layer)
    bg.paste(sharpened, (draw_x, draw_y), sharpened)
    return bg.convert("RGB")


# =============================================================================
# 에이전트 3: SEO · AEO · GEO 최적화 카피 및 4대 Q&A 생성 (Gemini API + 룰 폴백)
# =============================================================================
def build_seo_copy_and_qa(prod: dict) -> tuple[str, list[dict]]:
    """Gemini API 키가 있으면 AI 생성, 없어도 패키지 제조 표준 규칙으로 100% 완성"""
    name = prod["name"]
    dim = prod["dim_text"]
    mat = prod["material"]
    whole = prod["sub"]["whole"]
    half = prod["sub"]["half"]

    default_subcopy = f"외경 {dim} 규격의 고강도 {mat} 기성 패키지 — 매장 배달 포장에 최적화된 팩스타 직생산 제품"
    delivery_note = (
        "골판 합지 박스는 보관 부피를 줄이기 위해 펼친 상태로 출고되며, 접힘선을 따라 끼우는 간편 조립 구조로 매장에서 빠르게 완성할 수 있습니다."
        if mat == "합지"
        else f"{mat} 박스는 1차 접착 후 접힌 상태로 배송되어 별도 조립 없이 펼치기만 하면 즉시 포장할 수 있습니다."
    )
    tray_note = (
        f"외경 {dim} 규격 기준으로 한마리 메뉴는 {whole}, 반반 메뉴는 {half}가 정확히 호환됩니다."
        if half and "해당없음" not in half
        else f"외경 {dim} 규격 기준으로 {whole}가 정확히 호환됩니다."
    )

    fallback_qa = [
        {
            "q": f"원산지 표시와 {mat} 원단 강도는 어떤가요?",
            "a": f"식품용 {mat} 원단으로 제작되어 배달 적재 시 쉽게 찌그러지지 않으며, 박스 측면 원산지 표기란을 활용해 별도 스티커 없이 체크만으로 표기할 수 있습니다.",
        },
        {
            "q": f"이 박스({dim})에 정확히 맞는 호환 트레이는 무엇인가요?",
            "a": f"{tray_note} 트레이에 담아 포장하면 기름과 양념이 박스에 직접 닿는 것을 줄여 배달 품질을 유지할 수 있습니다.",
        },
        {
            "q": "배달 중 튀김이나 음식이 눅눅해지지 않나요?",
            "a": "뚜껑과 측면에 숨구멍이 설계되어 내부 수증기와 열기가 자연스럽게 배출됩니다. 배달 이동 중 쉽게 열리지 않는 결합 구조로 안정적인 포장이 가능합니다.",
        },
        {
            "q": "박스는 조립된 상태로 배송되나요?",
            "a": delivery_note,
        },
    ]

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[AGENT 3] GEMINI_API_KEY 미설정 — 팩스타 표준 SEO/Q&A 규칙 엔진으로 생성합니다.")
        return default_subcopy, fallback_qa

    try:
        from google import genai

        print("[AGENT 3] Gemini API로 B2B SEO·AEO·GEO 최적화 카피 및 Q&A 생성 중...")
        client = genai.Client(api_key=api_key)
        prompt = f"""
당신은 B2B 식품포장재 제조사 팩스타(FEXTA)의 수석 온라인 MD입니다.
아래 실제 상품 스펙을 바탕으로 검색엔진(SEO)과 생성형 AI(AEO/GEO)가 정확히 인용할 수 있는
간결하고 신뢰감 있는 한 줄 요약(subcopy)과 매장 사장님 대상 Q&A 4개를 순수 JSON으로만 반환하세요.
과장된 광고 수식어나 이모지는 절대 쓰지 마세요.

- 상품명: {name}
- 외경 규격: {dim}
- 재질/원단: {mat}
- 기본 입수량: {prod['quantity']}
- 한마리 호환 트레이: {whole}
- 반반 호환 트레이: {half}
- 전용 비닐봉투: {prod['sub']['bag']}

JSON 형식:
{{"subcopy": "45자 이내 핵심 요약", "qa": [{{"q": "질문1", "a": "답변1"}}, {{"q": "질문2", "a": "답변2"}}, {{"q": "질문3", "a": "답변3"}}, {{"q": "질문4", "a": "답변4"}}]}}
"""
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw_txt = re.sub(r"^```json\s*|\s*```$", "", (resp.text or "").strip())
        parsed = json.loads(raw_txt)
        subcopy = parsed.get("subcopy") or default_subcopy
        qa = parsed.get("qa") if isinstance(parsed.get("qa"), list) and len(parsed["qa"]) >= 4 else fallback_qa
        return subcopy, qa[:4]
    except Exception as exc:
        print(f"[WARN] Gemini API 호출 실패, 표준 규칙으로 자동 폴백: {exc}")
        return default_subcopy, fallback_qa


# =============================================================================
# 에이전트 4: 수석 디자이너 100% 인라인 카페24 HTML 조립기
# =============================================================================
def build_sub_placeholder_svg(kind: str, label: str) -> str:
    """자사몰 부자재 썸네일 검색 실패 시 깔끔한 미니멀 벡터 폴백"""
    safe = re.sub(r'[<>&"\']', "", label or "")[:18]
    if kind == "whole":
        body = '<rect x="42" y="48" width="96" height="64" rx="7" fill="#FFFFFF" stroke="#121211" stroke-width="2.2"/><text x="90" y="84" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="800" fill="#121211">ONE TRAY</text>'
    elif kind == "half":
        body = '<rect x="34" y="48" width="52" height="64" rx="6" fill="#FFFFFF" stroke="#121211" stroke-width="2.2"/><rect x="94" y="48" width="52" height="64" rx="6" fill="#FFFFFF" stroke="#121211" stroke-width="2.2"/><text x="60" y="84" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="800" fill="#121211">HALF</text><text x="120" y="84" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="800" fill="#121211">HALF</text>'
    else:
        body = '<path d="M56 44 H124 L130 116 H50 Z" fill="#FFFFFF" stroke="#121211" stroke-width="2.2"/><rect x="75" y="53" width="30" height="8" rx="4" fill="none" stroke="#121211" stroke-width="2"/><text x="90" y="92" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="800" fill="#121211">PE BAG</text>'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180"><rect width="180" height="180" fill="#F5F4F0"/>{body}<text x="90" y="146" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#4D4C48">{safe}</text></svg>'
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def render_cafe24_html(
    prod: dict,
    subcopy: str,
    qa_list: list[dict],
    main_img_src: str,
    detail_items: list[dict],
) -> str:
    """100% 인라인 스타일 · Product+FAQPage JSON-LD 포함 카페24 표준 HTML 생성"""
    name = prod["name"]
    dim_text = prod["dim_text"]
    material = prod["material"]
    quantity = prod["quantity"]
    usage = prod["usage"]
    sub = prod["sub"]
    thumbs = prod["sub_thumbs"]

    structured_data = [
        {
            "@context": "https://schema.org/",
            "@type": "Product",
            "name": name,
            "brand": {"@type": "Brand", "name": "팩스타 (FEXTA)"},
            "manufacturer": {
                "@type": "Organization",
                "name": "(주)팩스타",
                "address": "경기도 남양주시 진접읍 경복대로바람골길 42-100",
            },
            "material": material,
            "description": f"{name} · 외경 {dim_text} · 원단 {material} · {quantity} · {subcopy}",
            "additionalProperty": [
                {"@type": "PropertyValue", "name": "외경 규격 (W×D×H)", "value": dim_text},
                {"@type": "PropertyValue", "name": "재질/원단", "value": material},
                {"@type": "PropertyValue", "name": "기본 입수량", "value": quantity},
                {"@type": "PropertyValue", "name": "한마리 호환 트레이", "value": sub["whole"]},
                {"@type": "PropertyValue", "name": "반반 호환 트레이", "value": sub["half"]},
                {"@type": "PropertyValue", "name": "전용 비닐봉투", "value": sub["bag"]},
            ],
        },
        {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": item["q"], "acceptedAnswer": {"@type": "Answer", "text": item["a"]}}
                for item in qa_list
            ],
        },
    ]
    json_ld = f'<script type="application/ld+json">{json.dumps(structured_data, ensure_ascii=False)}</script>'

    # 2x2 Q&A 카드
    qa_rows = []
    for i in range(0, len(qa_list), 2):
        left = qa_list[i]
        right = qa_list[i + 1] if i + 1 < len(qa_list) else None
        right_html = (
            f"""
            <div style="background:#FAF9F5; border:1px solid #E5E2D9; border-top:3px solid #121211; padding:24px 22px; min-height:165px;">
              <div style="font-size:11px; font-weight:900; letter-spacing:0.12em; color:#C2410C; margin-bottom:10px;">CHECK POINT 0{i+2}</div>
              <div style="font-size:16px; font-weight:800; color:#121211; line-height:1.45; margin-bottom:10px; word-break:keep-all;">{right['q']}</div>
              <div style="font-size:13.5px; color:#52514E; line-height:1.7; word-break:keep-all;">{right['a']}</div>
            </div>"""
            if right
            else ""
        )
        qa_rows.append(
            f"""
        <tr>
          <td style="width:50%; padding:10px; vertical-align:top;">
            <div style="background:#FAF9F5; border:1px solid #E5E2D9; border-top:3px solid #121211; padding:24px 22px; min-height:165px;">
              <div style="font-size:11px; font-weight:900; letter-spacing:0.12em; color:#C2410C; margin-bottom:10px;">CHECK POINT 0{i+1}</div>
              <div style="font-size:16px; font-weight:800; color:#121211; line-height:1.45; margin-bottom:10px; word-break:keep-all;">{left['q']}</div>
              <div style="font-size:13.5px; color:#52514E; line-height:1.7; word-break:keep-all;">{left['a']}</div>
            </div>
          </td>
          <td style="width:50%; padding:10px; vertical-align:top;">{right_html}</td>
        </tr>"""
        )

    # 호환 부자재 카드 (해당없음 자동 제외)
    pair_cards = []
    if sub["whole"] and "해당없음" not in sub["whole"]:
        pair_cards.append({"tag": "한마리 포장용", "title": sub["whole"], "img": thumbs["whole"] or build_sub_placeholder_svg("whole", sub["whole"])})
    if sub["half"] and "해당없음" not in sub["half"]:
        pair_cards.append({"tag": "반반 포장용", "title": sub["half"], "img": thumbs["half"] or build_sub_placeholder_svg("half", sub["half"])})
    if sub["bag"] and "해당없음" not in sub["bag"]:
        pair_cards.append({"tag": "전용 비닐봉투", "title": sub["bag"], "img": thumbs["bag"] or build_sub_placeholder_svg("bag", sub["bag"])})

    col_w = f"{100 // len(pair_cards)}%" if pair_cards else "33%"
    pair_section = ""
    if pair_cards:
        cells = "".join(
            f"""
          <td style="width:{col_w}; background:#FFFFFF; border:1px solid #E2DFD7; padding:22px 16px; text-align:center; vertical-align:top;">
            <span style="display:inline-block; background:#121211; color:#FFFFFF; font-size:11px; font-weight:800; padding:4px 12px; margin-bottom:16px;">{c['tag']}</span>
            <div style="width:152px; height:152px; margin:0 auto 14px auto; background:#F7F6F2; border:1px solid #ECEAE4; overflow:hidden;">
              <img src="{c['img']}" alt="{name} 호환 {c['title']}" style="width:100%; height:100%; object-fit:contain; display:block;" />
            </div>
            <div style="font-size:15px; font-weight:900; color:#121211; margin-bottom:4px; word-break:keep-all;">{c['title']}</div>
            <div style="font-size:12px; color:#7A7872;">상단 추가구성에서 선택</div>
          </td>"""
            for c in pair_cards
        )
        pair_section = f"""
  <!-- [BLOCK 5] 03 · PAIR 호환 부자재 -->
  <div style="margin-top:68px; border:1px solid #121211; background:#FAF9F5; padding:34px 20px;">
    <div style="text-align:center; margin-bottom:26px;">
      <span style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; display:block; margin-bottom:6px;">03 &nbsp;·&nbsp; COMPATIBLE ADD-ON PAIR</span>
      <h2 style="margin:0 0 6px 0; font-size:24px; font-weight:900; color:#121211; letter-spacing:-0.03em;">함께 구매하면 좋은 호환 부자재</h2>
      <p style="margin:0; font-size:13.5px; color:#52514E;">상단 <strong>[추가구매상품]</strong> 옵션에서 호환 트레이와 비닐봉투를 한 번에 담아 주문하실 수 있습니다.</p>
    </div>
    <table style="width:100%; border-collapse:separate; border-spacing:12px 0;">
      <tbody><tr>{cells}</tr></tbody>
    </table>
  </div>"""

    # 04 · DETAIL 제품 고화질 누끼 갤러리
    detail_section = ""
    if detail_items:
        cards_html = "".join(
            f"""
    <div style="margin-bottom:32px; border:1px solid #E5E2D9; background:#FFFFFF;">
      <div style="background:#F7F6F2; padding:28px 20px; text-align:center;">
        <img src="{d['src']}" alt="{name} {d['caption']}" style="max-width:680px; width:100%; height:auto; display:inline-block;" />
      </div>
      <div style="padding:16px 24px; border-top:1px solid #E5E2D9; display:flex; align-items:center; justify-content:space-between; background:#FFFFFF;">
        <span style="font-size:11px; font-weight:900; letter-spacing:0.12em; color:#C2410C;">VIEW 0{idx+1}</span>
        <span style="font-size:14px; font-weight:700; color:#121211;">{d['caption']}</span>
      </div>
    </div>"""
            for idx, d in enumerate(detail_items)
        )
        detail_section = f"""
  <!-- [BLOCK 6] 04 · DETAIL 제품 고화질 실사·누끼 갤러리 -->
  <div style="margin-top:72px; padding-top:48px; border-top:2px solid #121211;">
    <div style="text-align:center; margin-bottom:36px;">
      <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:8px;">04 &nbsp;·&nbsp; PRODUCT DETAIL</div>
      <h2 style="margin:0 0 8px 0; font-size:28px; font-weight:900; color:#121211; letter-spacing:-0.03em;">제품 디테일 자세히 보기</h2>
      <p style="margin:0; font-size:14px; color:#6B6963;">{name} 실물 외관 · 인쇄 품질 · 마감 구조를 고화질로 확인하세요</p>
    </div>
    {cards_html}
  </div>"""

    # 공장 8대 공정 그리드
    factory_cells = []
    for idx, proc in enumerate(FACTORY_PROCESSES):
        f_url = prod["factory_urls"][idx] if idx < len(prod["factory_urls"]) else f"/web/upload/detail/common/factory/{proc['file']}"
        factory_cells.append(
            f"""
        <td style="width:25%; padding:8px; vertical-align:top; text-align:center;">
          <div style="background:#F5F4F0; border:1px solid #E2E0D8; overflow:hidden; margin-bottom:10px;">
            <img src="{f_url}" alt="팩스타 남양주 진접 공장 {proc['title']} 공정" style="width:100%; aspect-ratio:4/3; object-fit:cover; display:block;" />
          </div>
          <div style="font-size:11px; font-weight:800; color:#8A8883; margin-bottom:2px;">{proc['num']}</div>
          <div style="font-size:14px; font-weight:800; color:#121211; margin-bottom:4px;">{proc['title']}</div>
          <div style="font-size:12px; color:#52514E; line-height:1.45; word-break:keep-all;">{proc['desc']}</div>
        </td>"""
        )

    return f"""<!-- FEXTA 고화질 AI 누끼 · SEO/AEO/GEO 상세페이지 시작 -->
{json_ld}
<div style="max-width:860px; margin:0 auto; font-family:'Pretendard',-apple-system,BlinkMacSystemFont,sans-serif; color:#121211; line-height:1.65; letter-spacing:-0.02em; background:#FFFFFF;">

  <!-- [BLOCK 1] 최상단 상호인쇄 주문제작 공용 배너 -->
  <div style="margin-bottom:48px; background:#121211; border-bottom:3px solid #C2410C;">
    <a href="/board/inquiry/write.html" style="display:block; text-decoration:none; color:#FFFFFF; padding:26px 30px;">
      <table style="width:100%; border-collapse:collapse;">
        <tbody>
          <tr>
            <td style="vertical-align:middle; text-align:left;">
              <div style="font-size:11px; font-weight:800; letter-spacing:0.14em; color:#FDBA74; margin-bottom:6px;">FEXTA CUSTOM PACKAGING</div>
              <div style="font-size:20px; font-weight:900; color:#FFFFFF; letter-spacing:-0.02em;">매장 로고 상호인쇄 · 맞춤 패키지 주문제작</div>
              <div style="font-size:13px; color:#B4B2AC; margin-top:4px;">기성 박스 소량 인쇄부터 프랜차이즈 독판 제작까지 경기도 남양주 공장 직생산</div>
            </td>
            <td style="vertical-align:middle; text-align:right; width:210px;">
              <span style="display:inline-block; background:#FFFFFF; color:#121211; font-size:13px; font-weight:800; padding:12px 20px;">상호인쇄 주문제작 바로가기</span>
            </td>
          </tr>
        </tbody>
      </table>
    </a>
  </div>

  <!-- [BLOCK 2] 에디토리얼 스튜디오 히어로 -->
  <div style="text-align:center; padding:10px 16px 0 16px;">
    <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:10px;">FEXTA PACKAGING ARCHIVE</div>
    <h1 style="margin:0 0 12px 0; font-size:34px; font-weight:900; color:#121211; letter-spacing:-0.035em; line-height:1.25; word-break:keep-all;">{name}</h1>
    <p style="margin:0 auto 32px auto; max-width:600px; font-size:15.5px; color:#52514E; word-break:keep-all;">{subcopy}</p>
  </div>

  <div style="background:#F6F5F1; border:1px solid #E5E2D9; border-bottom:none; padding:42px 24px; text-align:center;">
    <img src="{main_img_src}" alt="{name} 외경 {dim_text} {material} 고화질 누끼 대표컷" style="max-width:520px; width:100%; height:auto; display:inline-block;" />
  </div>

  <table style="width:100%; border-collapse:collapse; background:#121211; color:#FFFFFF; margin-bottom:56px;">
    <tbody>
      <tr>
        <td style="width:36%; padding:22px 16px; border-right:1px solid #2C2B28; text-align:center;">
          <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#A6A49E; margin-bottom:5px;">외경 규격 (W×D×H)</div>
          <div style="font-size:18px; font-weight:900; color:#FFFFFF;">{dim_text}</div>
        </td>
        <td style="width:28%; padding:22px 16px; border-right:1px solid #2C2B28; text-align:center;">
          <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#A6A49E; margin-bottom:5px;">재질 / 원단</div>
          <div style="font-size:18px; font-weight:900; color:#FFFFFF;">{material}</div>
        </td>
        <td style="width:36%; padding:22px 16px; text-align:center;">
          <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#A6A49E; margin-bottom:5px;">기본 입수량</div>
          <div style="font-size:17px; font-weight:900; color:#FDBA74;">{quantity}</div>
        </td>
      </tr>
    </tbody>
  </table>

  <!-- [BLOCK 3] 01 · SPECIFICATION -->
  <div style="margin-top:56px;">
    <div style="text-align:center; margin-bottom:26px;">
      <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:8px;">01 &nbsp;·&nbsp; SPECIFICATION</div>
      <h2 style="margin:0; font-size:26px; font-weight:900; color:#121211; letter-spacing:-0.03em;">제품 상세 스펙</h2>
    </div>
    <table style="width:100%; border-collapse:collapse; border-top:2px solid #121211; border-bottom:1px solid #121211; font-size:15px; text-align:left;">
      <tbody>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="width:28%; padding:18px 20px; background:#FAF9F5; font-weight:800; color:#4D4C48;">외경 규격 (W×D×H)</th>
          <td style="padding:18px 22px; font-weight:900; font-size:18px; color:#121211;">{dim_text}</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 20px; background:#FAF9F5; font-weight:800; color:#4D4C48;">재질 / 원단</th>
          <td style="padding:18px 22px; font-weight:800; color:#121211;">{material}</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 20px; background:#FAF9F5; font-weight:800; color:#4D4C48;">기본 묶음 (입수량)</th>
          <td style="padding:18px 22px; font-weight:800; color:#C2410C;">{quantity}</td>
        </tr>
        <tr>
          <th style="padding:18px 20px; background:#FAF9F5; font-weight:800; color:#4D4C48;">권장 용도</th>
          <td style="padding:18px 22px; color:#292524;">{usage}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- [BLOCK 4] 02 · KEY POINTS & Q&A -->
  <div style="margin-top:68px;">
    <div style="text-align:center; margin-bottom:26px;">
      <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:8px;">02 &nbsp;·&nbsp; KEY POINTS &amp; Q&amp;A</div>
      <h2 style="margin:0 0 6px 0; font-size:26px; font-weight:900; color:#121211; letter-spacing:-0.03em;">주문 전 꼭 확인하는 네 가지 포인트</h2>
      <p style="margin:0; font-size:14px; color:#6B6963;">실제 매장 포장 환경 기준으로 가장 많이 문의하시는 내용을 정리했습니다</p>
    </div>
    <table style="width:100%; border-collapse:collapse;">
      <tbody>{"".join(qa_rows)}</tbody>
    </table>
  </div>

  {pair_section}
  {detail_section}

  <!-- [BLOCK 7] 05 · MADE BY FEXTA -->
  <div style="margin-top:72px; border-top:2px solid #121211; padding-top:46px;">
    <div style="text-align:center; margin-bottom:30px;">
      <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:8px;">05 &nbsp;·&nbsp; MADE BY FEXTA</div>
      <h2 style="margin:0 0 8px 0; font-size:28px; font-weight:900; color:#121211; letter-spacing:-0.03em;">만드는 곳을 확인하세요</h2>
      <p style="margin:0; font-size:14.5px; color:#6B6963;">팩스타가 경기도 남양주 진접 공장에서 직접 생산합니다</p>
    </div>
    <table style="width:100%; border-collapse:collapse;">
      <tbody>
        <tr>{"".join(factory_cells[:4])}</tr>
        <tr>{"".join(factory_cells[4:8])}</tr>
      </tbody>
    </table>
  </div>

  <!-- [BLOCK 8] 국제 품질·식품안전 인증 현황 4종 -->
  <div style="margin-top:52px; border-top:1px solid #121211; border-bottom:1px solid #E2DFD7;">
    <table style="width:100%; border-collapse:collapse; text-align:center;">
      <tbody>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="width:28%; padding:24px 12px; font-size:15px; font-weight:900; color:#121211; background:#FAF9F5; vertical-align:middle;">FSSC 22000</th>
          <td style="padding:24px 18px; font-size:14px; color:#292524; line-height:1.75;">
            <strong style="font-size:15px; color:#121211;">식품안전경영시스템 인증 · FSSC 22000 (Version 6)</strong><br />
            인증 범위: 식품산업용 종이박스의 생산(육안검사, 경화, 완제품 검사)<br />
            한국품질재단 · FC-1073601 &nbsp;|&nbsp; FSSC COID: KOR-1-1832-753001<br />
            인증 유효기간: 2026.10.06 ~ 2029.10.05
          </td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:24px 12px; font-size:15px; font-weight:900; color:#121211; background:#FAF9F5; vertical-align:middle;">FSC®</th>
          <td style="padding:24px 18px; font-size:14px; color:#292524; line-height:1.75;">
            <strong style="font-size:15px; color:#121211;">국제산림관리협의회 FSC® CoC 친환경 인증 보유 기업</strong><br />
            (FSC-C192781 · SCS-COC-010064)
          </td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:24px 12px; font-size:15px; font-weight:900; color:#121211; background:#FAF9F5; vertical-align:middle;">ISO 9001:2015</th>
          <td style="padding:24px 18px; font-size:14px; color:#292524; line-height:1.75;">
            <strong style="font-size:15px; color:#121211;">품질경영시스템 인증</strong><br />
            인증 범위: 포장재 및 포장용기의 설계/개발 및 생산<br />
            국제표준인증원 · ISCQM-2836 &nbsp;|&nbsp; 인증서 유효만료일: 2027.04.09
          </td>
        </tr>
        <tr>
          <th style="padding:24px 12px; font-size:15px; font-weight:900; color:#121211; background:#FAF9F5; vertical-align:middle;">ISO 14001:2015</th>
          <td style="padding:24px 18px; font-size:14px; color:#292524; line-height:1.75;">
            <strong style="font-size:15px; color:#121211;">환경경영시스템 인증</strong><br />
            인증 범위: 포장재 및 포장용기의 설계/개발 및 생산<br />
            국제표준인증원 · ISCEM-1512 &nbsp;|&nbsp; 인증서 유효만료일: 2027.04.09
          </td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- [BLOCK 9] 06 · CHECK 주문 전 확인 -->
  <div style="margin-top:72px; border-top:2px solid #121211; padding-top:46px; padding-bottom:28px;">
    <div style="text-align:center; margin-bottom:28px;">
      <div style="font-size:11px; font-weight:800; letter-spacing:0.18em; color:#C2410C; margin-bottom:8px;">06 &nbsp;·&nbsp; CHECK</div>
      <h2 style="margin:0; font-size:28px; font-weight:900; color:#121211; letter-spacing:-0.03em;">주문 전 확인</h2>
    </div>
    <table style="width:100%; border-collapse:collapse; border-top:2px solid #121211; border-bottom:1px solid #121211; text-align:center; font-size:14px;">
      <tbody>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="width:26%; padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">출고</th>
          <td style="padding:18px 16px; color:#292524;">오후 13:30 이전 결제 완료 시 당일 발송 · 롯데택배 · 보통 1~2일 소요</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">배송 형태</th>
          <td style="padding:18px 16px; color:#292524; line-height:1.85;">
            치킨박스 합지 : 펼친 상태로 배송 / 받은 뒤 조립<br />
            치킨박스 마닐라지 : 접힌 상태로 배송 / 펼치면 완성<br />
            트레이 : 완성 상태로 배송
          </td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">배송비</th>
          <td style="padding:18px 16px; color:#292524;">1회 주문 4,000원 · 산간·도서 지역은 추가 요금</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">샘플</th>
          <td style="padding:18px 16px; color:#292524;">샘플신청 메뉴에서 신청 · 비용과 신청 조건은 신청 페이지에서 확인해주세요.</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">로고 인쇄</th>
          <td style="padding:18px 16px; color:#292524;">상호 인쇄 제작은 쇼핑몰 상단에 <strong>주문제작</strong> 메뉴에서 확인해주세요</td>
        </tr>
        <tr style="border-bottom:1px solid #E5E2D9;">
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">교환·반품</th>
          <td style="padding:18px 16px; color:#292524;">수령 후 7일 이내 · 내피 포장을 개봉했거나 훼손된 경우 불가</td>
        </tr>
        <tr>
          <th style="padding:18px 12px; background:#FAF9F5; font-weight:800; color:#121211;">고객센터</th>
          <td style="padding:18px 16px; color:#292524; line-height:1.75;">
            070-4455-5945 / 카카오톡 채널 '팩스타'<br />
            상담시간 10:00~17:00 (점심시간 12:30 ~ 13:30)
          </td>
        </tr>
      </tbody>
    </table>
  </div>

</div>
<!-- FEXTA 고화질 AI 누끼 · SEO/AEO/GEO 상세페이지 끝 -->"""


# =============================================================================
# 메인 실행 파이프라인
# =============================================================================
def main():
    product_url = os.environ.get("PRODUCT_URL", "").strip()
    if not product_url:
        print("[ERROR] PRODUCT_URL 환경변수가 비어 있습니다.")
        sys.exit(1)

    material_override = os.environ.get("MATERIAL_OVERRIDE", "자동 판별").strip()
    bg_mode_raw = os.environ.get("CUTOUT_BG", "studio").strip().lower()
    bg_mode = "transparent" if "transparent" in bg_mode_raw else ("white" if "white" in bg_mode_raw else "studio")
    out_size = int(os.environ.get("OUTPUT_SIZE", "1200").strip() or "1200")
    detail_cutout_enabled = "누끼" in os.environ.get("DETAIL_CUTOUT_MODE", "누끼")

    ext = "png" if bg_mode == "transparent" else "jpg"
    save_fmt = "PNG" if bg_mode == "transparent" else "JPEG"

    out_dir = Path("output")
    img_dir = out_dir / "upload_images"
    skipped_dir = out_dir / "skipped_long_sheets"
    img_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)

    # 1. 에이전트 1 실행 (수집)
    prod = scrape_cafe24_product(product_url, material_override=material_override)
    ftp_folder = f"/web/upload/detail/product_{prod['product_no']}/"
    print(f"[AGENT 1 완료] 상품명: {prod['name']} | 규격: {prod['dim_text']} | 원단: {prod['material']}")

    # 2. 에이전트 2 실행 (고화질 AI 누끼)
    rembg_session = init_rembg_session()
    uploaded_files = []

    main_filename = f"02_product_cutout.{ext}"
    main_local_rel = f"upload_images/{main_filename}"
    main_ftp_path = f"{ftp_folder}{main_filename}"

    if prod["main_img_url"]:
        print(f"[AGENT 2] 대표 썸네일 고화질 AI 누끼 생성 중: {prod['main_img_url']}")
        raw_main = download_image(prod["main_img_url"])
        if raw_main:
            processed_main = create_studio_highres_cutout(raw_main, rembg_session, out_size=out_size, bg_mode=bg_mode)
            save_kwargs = {"quality": 95, "subsampling": 0} if save_fmt == "JPEG" else {}
            processed_main.save(img_dir / main_filename, format=save_fmt, **save_kwargs)
            uploaded_files.append((main_filename, main_ftp_path, f"대표 AI 누끼 ({out_size}×{out_size}px)"))
        else:
            main_ftp_path = prod["main_img_url"]
            main_local_rel = prod["main_img_url"]

    # 본문 상세 이미지 분석: 세로 긴 구형 통이미지(ratio > 2.0) 자동 차단 + 순수 제품 컷 고화질 처리
    detail_ftp_items = []
    detail_preview_items = []
    slot_idx = 0

    for raw_url in prod["detail_img_urls"]:
        if slot_idx >= 6:
            break
        raw_img = download_image(raw_url)
        if not raw_img:
            continue
        w, h = raw_img.size
        ratio = round(h / max(1, w), 2)

        if ratio > 2.0:
            # 세로 긴 구형 통이미지는 본문에서 자동 제외 (참고 폴더에만 보관)
            skip_name = f"blocked_long_{len(list(skipped_dir.iterdir())) + 1}.jpg"
            raw_img.convert("RGB").save(skipped_dir / skip_name, format="JPEG", quality=88)
            print(f"[AGENT 2] 구형 통이미지 자동 제외 ({w}×{h}, 비율 {ratio}) -> {skip_name}")
            continue

        if w < 220 or h < 220:
            continue

        slot_idx += 1
        d_file = f"03_detail_0{slot_idx}.{ext}"
        caption = DEFAULT_CAPTIONS[slot_idx - 1]

        if detail_cutout_enabled:
            print(f"[AGENT 2] 상세 컷 {slot_idx} AI 고화질 누끼 변환 중 ({w}×{h})...")
            final_img = create_studio_highres_cutout(raw_img, rembg_session, out_size=out_size, bg_mode=bg_mode)
            save_kwargs = {"quality": 95, "subsampling": 0} if save_fmt == "JPEG" else {}
            final_img.save(img_dir / d_file, format=save_fmt, **save_kwargs)
            uploaded_files.append((d_file, f"{ftp_folder}{d_file}", f"상세 컷 {slot_idx} AI 누끼 ({out_size}×{out_size}px)"))
        else:
            d_file = f"03_detail_0{slot_idx}.jpg"
            raw_img.convert("RGB").save(img_dir / d_file, format="JPEG", quality=95)
            uploaded_files.append((d_file, f"{ftp_folder}{d_file}", f"상세 컷 {slot_idx} 원본 고화질"))

        detail_ftp_items.append({"src": f"{ftp_folder}{d_file}", "caption": caption})
        detail_preview_items.append({"src": f"upload_images/{d_file}", "caption": caption})

    # 3. 에이전트 3 실행 (SEO·AEO·GEO 카피 + Q&A)
    subcopy, qa_list = build_seo_copy_and_qa(prod)

    # 4. 에이전트 4 실행 (카페24 붙여넣기용 HTML + PC 검수용 미리보기 HTML + 업로드 가이드)
    cafe24_html = render_cafe24_html(prod, subcopy, qa_list, main_ftp_path, detail_ftp_items)
    preview_body = render_cafe24_html(prod, subcopy, qa_list, main_local_rel, detail_preview_items)
    preview_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>[미리보기] {prod['name']} — FEXTA 상세페이지</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
  <style>body {{ margin:0; padding:24px 12px; background:#EAE8E2; }}</style>
</head>
<body>
  <div style="max-width:860px; margin:0 auto; background:#FFFFFF; box-shadow:0 12px 32px rgba(0,0,0,0.08);">
    {preview_body}
  </div>
</body>
</html>"""

    (out_dir / "cafe24_detail.html").write_text(cafe24_html, encoding="utf-8")
    (out_dir / "preview_check.html").write_text(preview_doc, encoding="utf-8")
    (out_dir / "extracted_data.json").write_text(
        json.dumps({"product": prod, "subcopy": subcopy, "qa": qa_list}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    guide_lines = [
        "======================================================================",
        "  FEXTA 카페24 상세페이지 적용 & WebFTP 업로드 가이드",
        "======================================================================",
        f"상품명       : {prod['name']} (No.{prod['product_no']})",
        f"규격 / 원단  : {prod['dim_text']} / {prod['material']} / {prod['quantity']}",
        f"호환 부자재  : 한마리({prod['sub']['whole']}) | 반반({prod['sub']['half']}) | 봉투({prod['sub']['bag']})",
        "",
        "[1단계] PC에서 먼저 결과 확인하기",
        "  - 압축 해제 후 'preview_check.html' 파일을 더블클릭하면 완성된 상세페이지를 눈으로 바로 확인할 수 있습니다.",
        "",
        "[2단계] 카페24 파일업로더(WebFTP) 이미지 업로드",
        "  - 카페24 관리자 > 디자인관리 > 디자인 부가기능 > 파일업로더(WebFTP)",
        f"  - 아래 폴더를 만들고 'upload_images' 폴더 안의 파일들을 이름 그대로 업로드하세요:",
        f"    ▶ 업로드 폴더: {ftp_folder}",
        "",
    ]
    for fname, fpath, desc in uploaded_files:
        guide_lines.append(f"    * {fname:<24} -> {fpath}  ({desc})")
    guide_lines += [
        "",
        "  ※ 호환 트레이·비닐봉투 썸네일 및 공장 8대 공정 이미지는 자사몰 기존 공용 URL을 재사용하므로 추가 업로드가 필요 없습니다.",
        "",
        "[3단계] 카페24 상품 상세설명 HTML 붙여넣기",
        "  - 카페24 상품 수정 > 상품상세설명 > [HTML] 탭 클릭",
        "  - 'cafe24_detail.html' 파일 내용을 전체 복사(Ctrl+A, Ctrl+C)하여 붙여넣기(Ctrl+V) 후 저장",
        "======================================================================",
    ]
    (out_dir / "UPLOAD_GUIDE.txt").write_text("\n".join(guide_lines), encoding="utf-8")
    print(f"[AGENT 4 완료] 생성된 이미지 {len(uploaded_files)}장 + cafe24_detail.html + preview_check.html 출력 완료!")


if __name__ == "__main__":
    main()
