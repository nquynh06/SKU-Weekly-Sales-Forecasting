#!/usr/bin/env python3
"""Vẽ workflow RF bằng matplotlib; không cần dot, mmdc hoặc mạng.

Chạy với Python đã có matplotlib. Chỉ xuất PNG và SVG cạnh script này.
Nguồn Mermaid ở random_forest_workflow.mmd là bản mô tả cùng quy trình.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})

COLORS = {
    "ink": "#0f172a",
    "muted": "#475569",
    "blue": ("#e0f2fe", "#0284c7"),
    "amber": ("#fef3c7", "#d97706"),
    "indigo": ("#e0e7ff", "#4f46e5"),
    "green": ("#dcfce7", "#16a34a"),
    "red": ("#fff1f2", "#e11d48"),
}
fig, ax = plt.subplots(figsize=(18, 18), facecolor="white")
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.012, top=0.995)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, width, height, title, body, color, fontsize=11.5):
    face, edge = COLORS[color]
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.009,rounding_size=0.009",
        linewidth=1.7, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y + height * 0.28, title, ha="center", va="center",
            fontsize=13.4, weight="bold", color=COLORS["ink"], zorder=4)
    ax.text(x, y - height * 0.12, body, ha="center", va="center",
            fontsize=fontsize, linespacing=1.35, color=COLORS["ink"], zorder=4)
    return {"top": (x, y + height / 2 + 0.009),
            "bottom": (x, y - height / 2 - 0.009),
            "left": (x - width / 2 - 0.009, y),
            "right": (x + width / 2 + 0.009, y)}


def arrow(start, end, color="#64748b", dashed=False, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=17,
        linewidth=1.8, color=color, linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle, zorder=2,
    ))


ax.text(0.5, 0.972, "RANDOM FOREST · DỰ BÁO LƯỢNG BÁN SKU–TUẦN",
        fontsize=23, weight="bold", ha="center", color=COLORS["ink"])
ax.text(0.5, 0.948, "Cuối tuần t → đúng tuần t+1  |  H=1  |  Gross completed units  |  Luồng local cần duyệt giả định",
        fontsize=13.1, ha="center", color=COLORS["muted"])

source = box(0.5, 0.892, 0.70, 0.065, "1. SNAPSHOT NGUỒN CSV",
             "SalesOrders · SalesOrderLines · ProductVariants; WeeklyCoverage tùy chọn\nKhông join giá, khuyến mãi, tồn kho, khách hàng hoặc returns", "blue")
gate = box(0.44, 0.797, 0.58, 0.065, "2. QUALITY GATE + LOG TRUY NGUYÊN",
           "Khóa / duplicate · FK · Quantity · trạng thái · ngày\nLỗi cứng: từ chối; loại trừ hợp lệ và giả định phải được ghi log", "amber")
fail = box(0.865, 0.797, 0.20, 0.065, "DỪNG KHI LỖI CỨNG",
           "DQ / failure diagnostics\nKhông ép tạo dự báo", "red", fontsize=10.6)
panel = box(0.5, 0.695, 0.82, 0.074, "3. PANEL VÀ ĐẶC TRƯNG ĐÚNG CUTOFF",
            "Thứ Hai; cohort từ first sale ≤ origin, không nhìn last sale tương lai\nNguồn đủ: tuần vắng = 0; nguồn thiếu = NaN; không bfill\nLag 1/2/4/8 · rolling 4/8 · zero-rate · age · calendar tuần đích", "blue")
arrow(source["bottom"], gate["top"])
arrow(gate["right"], fail["left"], color="#e11d48")
arrow(gate["bottom"], panel["top"])

for x in (0.255, 0.745):
    ax.add_patch(FancyBboxPatch(
        (x - 0.235, 0.178), 0.47, 0.456,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2, zorder=0))
ax.text(0.255, 0.616, "A. run · HUẤN LUYỆN / ĐÁNH GIÁ", fontsize=15,
        weight="bold", ha="center", color="#3730a3")
ax.text(0.745, 0.616, "B. predict · NẠP MODEL ĐÃ LƯU", fontsize=15,
        weight="bold", ha="center", color="#166534")

split = box(0.255, 0.559, 0.414, 0.066, "4A. SPLIT THEO TARGET_WEEK",
            "Train → 3 block validation mở rộng → test khóa\n2 ứng viên RF; chọn WAPE (MAE nếu actual = 0)\nOne-hot SKU chỉ fit trên train mỗi lần", "indigo", fontsize=11.0)
test = box(0.255, 0.453, 0.414, 0.066, "5A. TEST ĐỘC LẬP, CÙNG KEYS",
           "Một fit qua validation_end, không tune bằng test\nRF + MA4 + naïve; MAE / RMSE / WAPE\nMacro-SKU MAE · bias · coverage / fallback", "indigo", fontsize=11.0)
refit = box(0.255, 0.347, 0.414, 0.066, "6A. REFIT ĐỂ DEPLOY, SAU ĐÁNH GIÁ",
            "Cấu hình đã chọn; fit đến cuối snapshot\nSaved preprocessing + forest + contract\nKhông xem model deploy là model chấm điểm test", "indigo", fontsize=11.0)
artifacts = box(0.255, 0.237, 0.414, 0.070, "7A. ARTIFACTS / TRUY NGUYÊN",
                "model.joblib · manifest · DQ · weekly/features\nValidation / test predictions · metrics\nFeature importance; không mặc định thắng baseline", "green", fontsize=10.9)
load = box(0.745, 0.559, 0.414, 0.066, "4B. NẠP ARTIFACT ĐÁNG TIN CẬY",
           "Saved model + snapshot đến origin t\nĐối chiếu cutoff / contract / schema\nKhông dùng artifact đã fit sau origin", "green", fontsize=11.0)
route = box(0.745, 0.453, 0.414, 0.066, "5B. TẠO FEATURE CÙNG CODE run",
            "Đủ 8 tuần liên tục known → RF, kể cả SKU lạ\nNgắn / gappy → MA4 trên known last4\nKhông tuần known nào → unavailable, qhat trống", "amber", fontsize=10.9)
forecast = box(0.745, 0.347, 0.414, 0.066, "6B. XUẤT FORECAST + TRẠNG THÁI",
               "SKU_ID · origin · target · qhat + 2 baseline columns\nmodel / run · status · reason · as_of = origin + 7d\nCutoff event-time exclusive, không phải ingest time", "green", fontsize=10.8)
review = box(0.745, 0.237, 0.414, 0.070, "7B. RÀ SOÁT NGHIỆP VỤ / BI Ở BƯỚC SAU",
             "Người phụ trách kiểm lịch sử, chất lượng, kế hoạch\nKhông tự ra đơn mua hoặc tối ưu tồn kho\nĐối chiếu actual khi tuần đích kết thúc", "green", fontsize=11.0)
arrow(panel["bottom"], (0.255, 0.639))
arrow((0.255, 0.599), split["top"])
arrow(panel["bottom"], (0.745, 0.639))
arrow((0.745, 0.599), load["top"])
for left, right in ((split, test), (test, refit), (refit, artifacts),
                    (load, route), (route, forecast), (forecast, review)):
    arrow(left["bottom"], right["top"])
arrow(refit["right"], load["left"], color="#4f46e5", dashed=True,
      connectionstyle="arc3,rad=-0.12")
ax.text(0.502, 0.488, "saved\nmodel", fontsize=10.0, ha="center",
        va="center", color="#4f46e5", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})

box(0.5, 0.103, 0.93, 0.085, "RANH GIỚI VÀ GIẢ ĐỊNH CẦN PHÊ DUYỆT",
    "Observed sales ≠ nhu cầu không bị giới hạn tồn kho; gross units không trừ returns.\nFeed đủ / snapshot status / cohort / ngưỡng local chưa được chứng minh hoặc duyệt thay nhóm.\nKhông random split · không leakage · không gán missing thành zero · không tự tuyên bố cải thiện KPI.",
    "red", fontsize=12.0)
ax.text(0.5, 0.027, "Tài liệu tham chiếu: docs/random_forest_pipeline.md  •  Nguồn quy trình: random_forest_workflow.mmd",
        ha="center", fontsize=11.0, color=COLORS["muted"])

for suffix in ("svg", "png"):
    destination = HERE / f"random_forest_workflow.{suffix}"
    fig.savefig(destination, dpi=170, bbox_inches="tight", facecolor="white",
                metadata={"Description": "Workflow Random Forest dự báo lượng bán SKU–tuần; không phải kết quả benchmark."})
    if suffix == "svg":
        # Matplotlib inserts trailing spaces in path data; keep generated diffs clean.
        text = destination.read_text(encoding="utf-8")
        destination.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
    print(destination)
plt.close(fig)
