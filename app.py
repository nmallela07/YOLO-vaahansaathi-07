import os
import uuid
import csv
import subprocess
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response, send_file
from ultralytics import YOLO
from tqdm import tqdm
import imageio_ffmpeg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

MODEL_PATH = "new.onnx"

# ── Tuning constants ──────────────────────────────────────────────────────────
DEFAULT_CONF      = 0.15   # Low enough to catch motorcycles in CCTV footage
NMS_IOU           = 0.45  # IoU threshold for NMS (class-aware by default)
INFER_SIZE        = 416   # Fixed — must match ONNX export size
# In a squished 416×416 frame from a 16:9 source, width is compressed ~56% vs height.
# Motorcycles (tall+narrow in reality) therefore appear with w/h < ~0.65 in squish space.
# Cars (wide) remain w/h > 0.80. We use this to fix Car→Motorcycle misclassifications.
ASPECT_THRESHOLD  = 0.52  # boxes narrower than this (w/h in squish) → Motorcycle
MOTO_MAX_CONF     = 0.38  # only reclassify if Car confidence is below this

# ── Class names matching the actual 5-class ONNX model ───────────────────────
CLASS_NAMES = {
    0: 'Ambulance',
    1: 'Bus',
    2: 'Car',
    3: 'Motorcycle',
    4: 'Truck',
}

# Distinct BGR colours per class for bounding boxes
CLASS_COLORS = {
    0: (0,   165, 255),   # Ambulance  → orange
    1: (255,   0,   0),   # Bus        → blue
    2: (0,   255,   0),   # Car        → green
    3: (0,   255, 255),   # Motorcycle → yellow
    4: (0,     0, 255),   # Truck      → red
}


try:
    model = YOLO(MODEL_PATH, task='detect')
    if not model.names or len(model.names) == 0:
        model.names = CLASS_NAMES
    print(f"✅ Model loaded. Classes: {model.names}")
except Exception as e:
    model = None
    print(f"⚠️  Could not load model ({MODEL_PATH}): {e}")


# ── Post-processing: geometry-based Motorcycle reclassifier ─────────────────
def reclassify_motorcycle(cls_id: int, conf: float,
                          x1s: float, y1s: float,
                          x2s: float, y2s: float) -> int:
    """Return corrected class_id.

    In a squished 416×416 frame from a 16:9 video, the horizontal axis is
    compressed by ~56% relative to vertical. A motorcycle+rider (tall & narrow
    in reality) therefore produces a bounding box where width < height in the
    squish space. Cars (wide) have width ≥ height.

    If the model calls something "Car" but its shape is clearly narrow/tall,
    we promote it to "Motorcycle".
    """
    CAR_CLASS        = 2
    MOTORCYCLE_CLASS = 3

    if cls_id != CAR_CLASS or conf >= MOTO_MAX_CONF:
        return cls_id          # leave high-confidence Cars alone

    w = x2s - x1s
    h = y2s - y1s
    if h <= 0:
        return cls_id

    aspect = w / h
    if aspect < ASPECT_THRESHOLD:
        return MOTORCYCLE_CLASS
    return cls_id


# ── Helper: draw boxes ────────────────────────────────────────────────────────
def draw_boxes(image: np.ndarray, boxes, names: dict,
               orig_w: int, orig_h: int) -> np.ndarray:
    """Draw bounding boxes on *image* (original resolution).

    Boxes are in INFER_SIZE×INFER_SIZE squish space — scale back with ratio.
    Applies reclassify_motorcycle() before drawing.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for box in boxes:
        conf   = float(box.conf[0].item())
        x1s, y1s, x2s, y2s = box.xyxy[0].tolist()
        cls_id = reclassify_motorcycle(
            int(box.cls[0].item()), conf, x1s, y1s, x2s, y2s
        )

        # Scale from INFER_SIZE×INFER_SIZE → original resolution
        x1 = int(x1s * orig_w / INFER_SIZE)
        y1 = int(y1s * orig_h / INFER_SIZE)
        x2 = int(x2s * orig_w / INFER_SIZE)
        y2 = int(y2s * orig_h / INFER_SIZE)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color      = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label      = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        thickness  = max(2, int(min(w, h) / 300))
        font_scale = max(0.4, min(w, h) / 1000)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(img, (x1, label_y1),
                      (x1 + tw + 4, label_y1 + th + baseline + 4), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return img


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve result files with HTTP range-request support (required for <video>)."""
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    file_size    = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    ext = filename.rsplit('.', 1)[-1].lower()
    mime_map = {
        'mp4': 'video/mp4', 'avi': 'video/x-msvideo',
        'mov': 'video/quicktime', 'mkv': 'video/x-matroska',
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
        'csv': 'text/csv',
    }
    mime_type = mime_map.get(ext, 'application/octet-stream')

    if not range_header:
        return send_file(file_path, mimetype=mime_type)

    byte_range = range_header.replace('bytes=', '').strip()
    parts  = byte_range.split('-')
    start  = int(parts[0]) if parts[0] else 0
    end    = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    end    = min(end, file_size - 1)
    length = end - start + 1

    def generate_chunk(path, offset, size, chunk=65536):
        with open(path, 'rb') as f:
            f.seek(offset)
            remaining = size
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    resp = Response(generate_chunk(file_path, start, length),
                    status=206, mimetype=mime_type, direct_passthrough=True)
    resp.headers['Content-Range']  = f'bytes {start}-{end}/{file_size}'
    resp.headers['Accept-Ranges']  = 'bytes'
    resp.headers['Content-Length'] = str(length)
    resp.headers['Cache-Control']  = 'no-cache'
    return resp


@app.route('/upload', methods=['POST'])
def upload():
    if not model:
        return jsonify({'error': f'Model not found: {MODEL_PATH}'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read confidence threshold from UI slider
    try:
        conf_thresh = float(request.form.get('conf', DEFAULT_CONF))
        conf_thresh = max(0.05, min(0.95, conf_thresh))
    except (TypeError, ValueError):
        conf_thresh = DEFAULT_CONF

    ext      = file.filename.rsplit('.', 1)[-1].lower()
    run_id   = uuid.uuid4().hex
    filename = f"{run_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    project_path = os.path.abspath(app.config['RESULT_FOLDER'])
    out_dir      = os.path.join(project_path, run_id)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "detections.csv")
    is_video = ext in ['mp4', 'avi', 'mov', 'mkv']
    names    = model.names or CLASS_NAMES

    try:
        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame_id', 'class_name', 'confidence',
                                 'x1', 'y1', 'x2', 'y2'])

            if is_video:
                cap    = cv2.VideoCapture(filepath)
                fps    = cap.get(cv2.CAP_PROP_FPS)
                fps    = fps if fps and fps == fps else 25.0
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out_media  = os.path.join(out_dir, f"{run_id}_yolo.mp4")
                fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(out_media, fourcc, fps, (orig_w, orig_h))

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_id     = 0

                with tqdm(total=total_frames, desc="Inferencing video") as pbar:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Squish to 416×416 — this is what the model expects
                        frame_sq = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

                        r = model.predict(
                            source=frame_sq,
                            imgsz=INFER_SIZE,
                            conf=conf_thresh,
                            iou=NMS_IOU,
                            verbose=False
                        )[0]

                        annotated = draw_boxes(frame, r.boxes, names, orig_w, orig_h)
                        vid_writer.write(annotated)

                        for box in r.boxes:
                            conf     = float(box.conf[0].item())
                            x1s, y1s, x2s, y2s = box.xyxy[0].tolist()
                            cls_id   = reclassify_motorcycle(
                                int(box.cls[0].item()), conf, x1s, y1s, x2s, y2s
                            )
                            cls_name = names.get(cls_id, str(cls_id))
                            x1 = int(x1s * orig_w / INFER_SIZE)
                            y1 = int(y1s * orig_h / INFER_SIZE)
                            x2 = int(x2s * orig_w / INFER_SIZE)
                            y2 = int(y2s * orig_h / INFER_SIZE)
                            csv_writer.writerow(
                                [frame_id, cls_name, f"{conf:.2f}",
                                 x1, y1, x2, y2]
                            )

                        frame_id += 1
                        pbar.update(1)

                cap.release()
                vid_writer.release()

            else:
                out_media    = os.path.join(out_dir, f"{run_id}.jpg")
                frame        = cv2.imread(filepath)

                frame_sq = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
                r = model.predict(
                    source=frame_sq,
                    imgsz=INFER_SIZE,
                    conf=conf_thresh,
                    iou=NMS_IOU,
                    verbose=False
                )[0]

                annotated = draw_boxes(frame, r.boxes, names, orig_w, orig_h)
                cv2.imwrite(out_media, annotated)

                for box in r.boxes:
                    conf     = float(box.conf[0].item())
                    x1s, y1s, x2s, y2s = box.xyxy[0].tolist()
                    cls_id   = reclassify_motorcycle(
                        int(box.cls[0].item()), conf, x1s, y1s, x2s, y2s
                    )
                    cls_name = names.get(cls_id, str(cls_id))
                    x1 = int(x1s * orig_w / INFER_SIZE)
                    y1 = int(y1s * orig_h / INFER_SIZE)
                    x2 = int(x2s * orig_w / INFER_SIZE)
                    y2 = int(y2s * orig_h / INFER_SIZE)
                    csv_writer.writerow(
                        [0, cls_name, f"{conf:.2f}", x1, y1, x2, y2]
                    )

    except Exception as e:
        return jsonify({'error': f'Inference error: {str(e)}'}), 500

    # ── Re-encode to browser-compatible H.264 ─────────────────────────────────
    if is_video:
        temp_media = out_media + ".temp.mp4"
        final_mp4  = os.path.join(out_dir, f"{run_id}.mp4")
        os.rename(out_media, temp_media)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y',
            '-r', f'{fps:.6f}',
            '-i', temp_media,
            '-vcodec', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-r', f'{fps:.6f}',
            '-pix_fmt', 'yuv420p',
            final_mp4
        ]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_media)
            saved_media = final_mp4
        except Exception as e:
            os.rename(temp_media, out_media)
            saved_media = out_media
            print(f"FFmpeg error (falling back to raw output): {e}")
    else:
        saved_media = out_media

    if not os.path.exists(saved_media):
        return jsonify({'error': 'Inference produced no output file.'}), 500

    result_file = os.path.basename(saved_media)
    file_type   = "video" if is_video else "image"

    return jsonify({
        'url':     f"/static/results/{run_id}/{result_file}",
        'csv_url': f"/static/results/{run_id}/detections.csv",
        'type':    file_type,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)