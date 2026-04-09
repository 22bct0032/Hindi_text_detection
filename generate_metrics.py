"""
Comprehensive Evaluation Script — generate_metrics.py
Calculates: Precision, Recall, F1, PSNR, DRD, RMSE + generates comparison graphs & confusion matrices.
Runs on: dataSet (model comparison) and OCR_dataset (angle-wise)
"""
import os, sys, io, csv, json, re, math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Ground truth word counts
# ============================================================
ACTUAL_DATASET = {}
for i in range(1, 12):   ACTUAL_DATASET[str(i)] = 124
for i in range(12, 22):  ACTUAL_DATASET[str(i)] = 124
for i in range(22, 32):  ACTUAL_DATASET[str(i)] = 124
for i in range(32, 42):  ACTUAL_DATASET[str(i)] = 125
for i in range(42, 52):  ACTUAL_DATASET[str(i)] = 122
for i in range(52, 62):  ACTUAL_DATASET[str(i)] = 126
for i in range(62, 72):  ACTUAL_DATASET[str(i)] = 121
for i in range(72, 82):  ACTUAL_DATASET[str(i)] = 121
for i in range(82, 92):  ACTUAL_DATASET[str(i)] = 121
for i in range(92, 98):  ACTUAL_DATASET[str(i)] = 122
ACTUAL_DATASET.update({"100": 122, "101": 122, "102": 124, "103": 124, "104": 124,
    "105": 124, "106": 124, "107": 124, "108": 124, "109": 124, "110": 124, "111": 124})

ACTUAL_DS2 = {"1": 124, "2": 124, "3": 126, "4": 125, "5": 122,
              "6": 126, "7": 122, "8": 121, "9": 121, "10": 121, "11": 124}

# ============================================================
# Helper: Calculate metrics from actual & detected counts
# ============================================================
def calc_metrics(actual_list, detected_list):
    """Calculate Precision, Recall, F1, RMSE from paired actual/detected counts."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    squared_errors = []

    for actual, detected in zip(actual_list, detected_list):
        tp = min(actual, detected)
        fp = max(0, detected - actual)
        fn = max(0, actual - detected)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        squared_errors.append((detected - actual) ** 2)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else 0
    mae = sum(abs(d - a) for a, d in zip(actual_list, detected_list)) / len(actual_list)

    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn
    }


def calc_psnr(original_path):
    """Calculate PSNR between original image and its binarized version."""
    img = cv2.imread(original_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 15, 10)
    mse = np.mean((gray.astype(float) - binary.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    psnr = 10 * math.log10(255.0 ** 2 / mse)
    return round(psnr, 2)


def calc_drd(original_path):
    """Calculate DRD (Distance Reciprocal Distortion) for binarization quality."""
    img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Ground truth: Otsu binarization (reference)
    _, gt = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Test: Adaptive threshold (what our algorithm uses)
    test = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 15, 10)

    # Find flipped pixels
    diff = (gt != test).astype(np.float64)
    num_flipped = np.sum(diff)
    if num_flipped == 0:
        return 0.0

    # DRD weight mask (5x5 normalized)
    wm = np.zeros((5, 5), dtype=np.float64)
    for i in range(5):
        for j in range(5):
            dist = math.sqrt((i - 2) ** 2 + (j - 2) ** 2)
            wm[i, j] = 1.0 / max(dist, 0.001)
    wm[2, 2] = 0
    wm /= np.sum(wm)

    # Calculate DRD
    h, w = gt.shape
    gt_norm = gt.astype(np.float64) / 255.0
    pad_gt = np.pad(gt_norm, 2, mode='edge')

    drd_sum = 0.0
    ys, xs = np.where(diff > 0)
    # Sample for speed (max 5000 pixels)
    if len(ys) > 5000:
        idx = np.random.choice(len(ys), 5000, replace=False)
        ys, xs = ys[idx], xs[idx]
        scale = num_flipped / 5000
    else:
        scale = 1.0

    for y, x in zip(ys, xs):
        block = pad_gt[y:y + 5, x:x + 5]
        diff_val = gt_norm[y, x]  # flipped pixel value
        distortion = np.sum(np.abs(block - diff_val) * wm)
        drd_sum += distortion

    drd_sum *= scale

    # Normalize by number of non-uniform blocks (8x8)
    bh, bw = h // 8, w // 8
    nubn = max(1, bh * bw)
    drd = drd_sum / nubn
    return round(drd, 4)


# ============================================================
# Style setup
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'text.color': '#eee',
    'axes.grid': True,
    'grid.color': '#333',
    'grid.alpha': 0.3,
    'font.size': 11,
})

COLORS = {
    'Our Algorithm': '#00d4ff',
    'EasyOCR': '#ff6b6b',
    'PaddleOCR': '#ffd93d',
    'Tesseract': '#6bcb77',
}


# ============================================================
# PART 1: dataSet — Model Comparison Metrics
# ============================================================
print("=" * 60)
print("  PART 1: dataSet — Model Comparison Metrics")
print("=" * 60)

# Read model_comparison.csv
models = {'Our Algorithm': [], 'EasyOCR': [], 'PaddleOCR': [], 'Tesseract': []}
actual_list = []
image_names = []

with open("outputs/model_comparison.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row[0] or 'Overall' in row[0] or 'MSE' in row[0] or 'MAE' in row[0] or 'Median' in row[0]:
            continue
        image_names.append(row[0])
        actual_list.append(int(row[1]))
        models['Our Algorithm'].append(int(row[2]))
        models['EasyOCR'].append(int(row[5]))
        models['PaddleOCR'].append(int(row[8]))
        models['Tesseract'].append(int(row[11]))

print(f"  Loaded {len(image_names)} images from model_comparison.csv")

# Calculate metrics for each model
all_metrics = {}
for name, detected_list in models.items():
    m = calc_metrics(actual_list, detected_list)
    all_metrics[name] = m
    print(f"\n  {name}:")
    print(f"    Precision: {m['precision']}%  |  Recall: {m['recall']}%  |  F1: {m['f1']}%")
    print(f"    RMSE: {m['rmse']}  |  MAE: {m['mae']}")
    print(f"    TP: {m['tp']}  |  FP: {m['fp']}  |  FN: {m['fn']}")

# Calculate PSNR & DRD (sample 10 images for speed)
print("\n  Calculating PSNR & DRD (sampling images)...")
psnr_values = []
drd_values = []
sample_images = [os.path.join("dataSet", f) for f in os.listdir("dataSet")
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:20]

for img_path in sample_images:
    p = calc_psnr(img_path)
    d = calc_drd(img_path)
    if p is not None:
        psnr_values.append(p)
    if d is not None:
        drd_values.append(d)

avg_psnr = round(np.mean(psnr_values), 2) if psnr_values else 0
avg_drd = round(np.mean(drd_values), 4) if drd_values else 0
print(f"  Average PSNR: {avg_psnr} dB")
print(f"  Average DRD: {avg_drd}")

# ── Save metrics CSV ──
metrics_csv = os.path.join(OUTPUT_DIR, "dataSet_metrics.csv")
with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Precision(%)', 'Recall(%)', 'F1-Score(%)', 'RMSE', 'MAE', 'TP', 'FP', 'FN'])
    for name in ['Our Algorithm', 'EasyOCR', 'PaddleOCR', 'Tesseract']:
        m = all_metrics[name]
        writer.writerow([name, m['precision'], m['recall'], m['f1'], m['rmse'], m['mae'], m['tp'], m['fp'], m['fn']])
    writer.writerow([])
    writer.writerow(['Image Quality Metrics'])
    writer.writerow(['PSNR (dB)', avg_psnr])
    writer.writerow(['DRD', avg_drd])
print(f"  Saved: {metrics_csv}")


# ── Graph 1: Precision, Recall, F1 Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(12, 6))
model_names = ['Our Algorithm', 'EasyOCR', 'PaddleOCR', 'Tesseract']
x = np.arange(len(model_names))
width = 0.25

prec = [all_metrics[m]['precision'] for m in model_names]
rec = [all_metrics[m]['recall'] for m in model_names]
f1s = [all_metrics[m]['f1'] for m in model_names]

bars1 = ax.bar(x - width, prec, width, label='Precision', color='#00d4ff', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, rec, width, label='Recall', color='#ff6b6b', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#ffd93d', edgecolor='white', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, color='white')

ax.set_xlabel('Model')
ax.set_ylabel('Score (%)')
ax.set_title('Precision, Recall & F1-Score — Model Comparison (dataSet)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 110)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_precision_recall_f1.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 1_precision_recall_f1.png")


# ── Graph 2: RMSE Comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
rmse_vals = [all_metrics[m]['rmse'] for m in model_names]
colors = [COLORS[m] for m in model_names]
bars = ax.bar(model_names, rmse_vals, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, rmse_vals):
    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11, fontweight='bold', color='white')
ax.set_ylabel('RMSE (lower is better)')
ax.set_title('RMSE — Model Comparison (dataSet)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_rmse_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 2_rmse_comparison.png")


# ── Graph 3: Detection Count per Image (line graph, first 30 images) ──
fig, ax = plt.subplots(figsize=(16, 6))
x_imgs = range(min(30, len(image_names)))
ax.plot(x_imgs, actual_list[:30], 'w--', linewidth=2, label='Actual', alpha=0.8)
for name in model_names:
    ax.plot(x_imgs, models[name][:30], color=COLORS[name], linewidth=1.5, label=name, alpha=0.85)
ax.set_xlabel('Image Index')
ax.set_ylabel('Word Count')
ax.set_title('Detected vs Actual Word Count (First 30 Images)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_detection_line_chart.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 3_detection_line_chart.png")


# ── Graph 4: Confusion Matrices ──
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, name in enumerate(model_names):
    m = all_metrics[name]
    cm = np.array([[m['tp'], m['fn']], [m['fp'], 0]])
    ax = axes[idx]
    im = ax.imshow(cm, cmap='Blues' if name == 'Our Algorithm' else 'Reds' if name == 'EasyOCR'
                   else 'YlOrBr' if name == 'PaddleOCR' else 'Greens', aspect='auto')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                    fontweight='bold', color=text_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=9)
    ax.set_yticklabels(['Actual+', 'Actual-'], fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_title(f'{name}\nF1={m["f1"]}%', fontsize=11, fontweight='bold')

fig.suptitle('Confusion Matrices — Model Comparison (dataSet)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 4_confusion_matrices.png")


# ── Graph 5: PSNR & DRD Bar ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(['Our Preprocessing'], [avg_psnr], color='#00d4ff', width=0.4, edgecolor='white')
ax1.annotate(f'{avg_psnr} dB', xy=(0, avg_psnr), xytext=(0, 5), textcoords="offset points",
             ha='center', fontsize=13, fontweight='bold', color='white')
ax1.set_ylabel('PSNR (dB) — Higher is Better')
ax1.set_title('PSNR (Image Quality)', fontsize=12, fontweight='bold')

ax2.bar(['Our Preprocessing'], [avg_drd], color='#ff6b6b', width=0.4, edgecolor='white')
ax2.annotate(f'{avg_drd}', xy=(0, avg_drd), xytext=(0, 5), textcoords="offset points",
             ha='center', fontsize=13, fontweight='bold', color='white')
ax2.set_ylabel('DRD — Lower is Better')
ax2.set_title('DRD (Binarization Quality)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "5_psnr_drd.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 5_psnr_drd.png")


# ── Graph 6: Overall Accuracy Comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
accs = []
for name in model_names:
    m = all_metrics[name]
    total_abs_err = sum(abs(d - a) for a, d in zip(actual_list, models[name]))
    total_actual = sum(actual_list)
    acc = round(100 - total_abs_err / total_actual * 100, 2)
    accs.append(acc)

bars = ax.bar(model_names, accs, color=[COLORS[m] for m in model_names],
              edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, accs):
    ax.annotate(f'{val}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=13, fontweight='bold', color='white')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Overall Detection Accuracy — Model Comparison (dataSet)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_accuracy_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 6_accuracy_comparison.png")


# ============================================================
# PART 2: OCR_dataset — Angle-wise Metrics
# ============================================================
print(f"\n{'='*60}")
print("  PART 2: OCR_dataset — Angle-wise Metrics")
print("=" * 60)

OCR_OUTPUT = "outputs/final_improved_OCR_dataset"
angle_data = defaultdict(lambda: {'actual': [], 'detected': []})

subfolders = sorted([d for d in os.listdir(OCR_OUTPUT) if os.path.isdir(os.path.join(OCR_OUTPUT, d))])

for sf in subfolders:
    sf_path = os.path.join(OCR_OUTPUT, sf)
    for jf in os.listdir(sf_path):
        if not jf.endswith('.json'):
            continue
        with open(os.path.join(sf_path, jf), 'r', encoding='utf-8') as f:
            data = json.load(f)

        detected = data.get('total_boxes', 0)
        m = re.match(r'page(\d+)_', jf)
        page_id = m.group(1) if m else jf.replace('.json', '')
        if page_id not in ACTUAL_DS2:
            continue

        actual = ACTUAL_DS2[page_id]
        angle_data[sf]['actual'].append(actual)
        angle_data[sf]['detected'].append(detected)

# Calculate metrics per angle range
angle_metrics = {}
for sf in sorted(angle_data.keys()):
    d = angle_data[sf]
    m = calc_metrics(d['actual'], d['detected'])
    angle_metrics[sf] = m
    print(f"  {sf}: P={m['precision']}% R={m['recall']}% F1={m['f1']}% RMSE={m['rmse']}")

# Calculate PSNR & DRD for OCR_dataset
print("\n  Calculating PSNR & DRD for OCR_dataset...")
ocr_psnr_per_angle = {}
ocr_drd_per_angle = {}
for sf in sorted(angle_data.keys()):
    sf_input = os.path.join("OCR_dataset", sf)
    if not os.path.isdir(sf_input):
        continue
    imgs = [f for f in os.listdir(sf_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    psnrs, drds = [], []
    for img in imgs:
        p = calc_psnr(os.path.join(sf_input, img))
        d = calc_drd(os.path.join(sf_input, img))
        if p is not None: psnrs.append(p)
        if d is not None: drds.append(d)
    ocr_psnr_per_angle[sf] = round(np.mean(psnrs), 2) if psnrs else 0
    ocr_drd_per_angle[sf] = round(np.mean(drds), 4) if drds else 0
    print(f"    {sf}: PSNR={ocr_psnr_per_angle[sf]} dB, DRD={ocr_drd_per_angle[sf]}")

# ── Save OCR_dataset metrics CSV ──
ocr_csv = os.path.join(OUTPUT_DIR, "OCR_dataset_metrics.csv")
with open(ocr_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Angle Range', 'Images', 'Precision(%)', 'Recall(%)', 'F1-Score(%)',
                      'RMSE', 'MAE', 'TP', 'FP', 'FN', 'PSNR(dB)', 'DRD'])
    for sf in sorted(angle_metrics.keys()):
        m = angle_metrics[sf]
        n = len(angle_data[sf]['actual'])
        writer.writerow([sf, n, m['precision'], m['recall'], m['f1'], m['rmse'], m['mae'],
                          m['tp'], m['fp'], m['fn'],
                          ocr_psnr_per_angle.get(sf, ''), ocr_drd_per_angle.get(sf, '')])
print(f"  Saved: {ocr_csv}")


# ── Graph 7: Angle-wise F1, Precision, Recall ──
fig, ax = plt.subplots(figsize=(14, 6))
angle_names = sorted(angle_metrics.keys())
x = np.arange(len(angle_names))
width = 0.25

prec = [angle_metrics[a]['precision'] for a in angle_names]
rec = [angle_metrics[a]['recall'] for a in angle_names]
f1s = [angle_metrics[a]['f1'] for a in angle_names]

ax.bar(x - width, prec, width, label='Precision', color='#00d4ff', edgecolor='white', linewidth=0.5)
ax.bar(x, rec, width, label='Recall', color='#ff6b6b', edgecolor='white', linewidth=0.5)
ax.bar(x + width, f1s, width, label='F1-Score', color='#ffd93d', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Angle Range')
ax.set_ylabel('Score (%)')
ax.set_title('Precision, Recall & F1-Score — Angle-wise (OCR_dataset)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([a.replace('_degree', '°').replace('_', '-') for a in angle_names], rotation=30, ha='right')
ax.set_ylim(90, 102)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_angle_precision_recall_f1.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 7_angle_precision_recall_f1.png")


# ── Graph 8: Angle-wise RMSE ──
fig, ax = plt.subplots(figsize=(12, 5))
rmse_vals = [angle_metrics[a]['rmse'] for a in angle_names]
bars = ax.bar(range(len(angle_names)), rmse_vals, color='#e94560', edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, rmse_vals):
    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold', color='white')
ax.set_xticks(range(len(angle_names)))
ax.set_xticklabels([a.replace('_degree', '°').replace('_', '-') for a in angle_names], rotation=30, ha='right')
ax.set_ylabel('RMSE')
ax.set_title('RMSE — Angle-wise (OCR_dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "8_angle_rmse.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 8_angle_rmse.png")


# ── Graph 9: Angle-wise Accuracy Line ──
fig, ax = plt.subplots(figsize=(12, 5))
acc_vals = []
for a in angle_names:
    d = angle_data[a]
    total_err = sum(abs(det - act) for act, det in zip(d['actual'], d['detected']))
    total_act = sum(d['actual'])
    acc_vals.append(round(100 - total_err / total_act * 100, 2))

ax.plot(range(len(angle_names)), acc_vals, 'o-', color='#00d4ff', linewidth=2.5, markersize=8)
for i, val in enumerate(acc_vals):
    ax.annotate(f'{val}%', xy=(i, val), xytext=(0, 10), textcoords="offset points",
                ha='center', fontsize=10, fontweight='bold', color='white')
ax.set_xticks(range(len(angle_names)))
ax.set_xticklabels([a.replace('_degree', '°').replace('_', '-') for a in angle_names], rotation=30, ha='right')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Detection Accuracy across Angles (OCR_dataset)', fontsize=14, fontweight='bold')
ax.set_ylim(90, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "9_angle_accuracy_line.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 9_angle_accuracy_line.png")


# ── Graph 10: Angle-wise PSNR & DRD ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
psnr_vals = [ocr_psnr_per_angle.get(a, 0) for a in angle_names]
drd_vals = [ocr_drd_per_angle.get(a, 0) for a in angle_names]

ax1.bar(range(len(angle_names)), psnr_vals, color='#00d4ff', edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(angle_names)))
ax1.set_xticklabels([a.replace('_degree', '°').replace('_', '-') for a in angle_names], rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('PSNR — Angle-wise', fontsize=12, fontweight='bold')

ax2.bar(range(len(angle_names)), drd_vals, color='#ff6b6b', edgecolor='white', linewidth=0.5)
ax2.set_xticks(range(len(angle_names)))
ax2.set_xticklabels([a.replace('_degree', '°').replace('_', '-') for a in angle_names], rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('DRD')
ax2.set_title('DRD — Angle-wise', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_angle_psnr_drd.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 10_angle_psnr_drd.png")


# ── Graph 11: Angle-wise Confusion Matrix ──
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for idx, sf in enumerate(angle_names):
    m = angle_metrics[sf]
    cm = np.array([[m['tp'], m['fn']], [m['fp'], 0]])
    ax = axes[idx // 4][idx % 4]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    fontweight='bold', color=text_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pos', 'Neg'], fontsize=8)
    ax.set_yticklabels(['Act+', 'Act-'], fontsize=8)
    label = sf.replace('_degree', '°').replace('_', '-')
    ax.set_title(f'{label}\nF1={m["f1"]}%', fontsize=10, fontweight='bold')

fig.suptitle('Confusion Matrices — Angle-wise (OCR_dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_angle_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 11_angle_confusion_matrices.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  ALL DONE! Output: {OUTPUT_DIR}/")
print(f"{'='*60}")
print(f"\n  Files generated:")
print(f"    CSVs:")
print(f"      - dataSet_metrics.csv")
print(f"      - OCR_dataset_metrics.csv")
print(f"    Graphs:")
print(f"      - 1_precision_recall_f1.png")
print(f"      - 2_rmse_comparison.png")
print(f"      - 3_detection_line_chart.png")
print(f"      - 4_confusion_matrices.png")
print(f"      - 5_psnr_drd.png")
print(f"      - 6_accuracy_comparison.png")
print(f"      - 7_angle_precision_recall_f1.png")
print(f"      - 8_angle_rmse.png")
print(f"      - 9_angle_accuracy_line.png")
print(f"      - 10_angle_psnr_drd.png")
print(f"      - 11_angle_confusion_matrices.png")
print(f"{'='*60}")
