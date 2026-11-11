"""HW3: COCO YOLO detection and per-object Canny edge detection on two photos.
Run: python HW3_YOLO_Canny.py --photo1 photo1.jpg --photo2 photo2.jpg
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics


def read_image(path):
    # imdecode supports Windows paths containing Korean characters.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def analyze_photo(model, image_path, photo_index, out_dir, args):
    image = read_image(image_path)
    height, width = image.shape[:2]
    # Ultralytics Boxes.xyxy returns coordinates mapped back to original image size.
    result = model.predict(source=image, conf=args.conf, iou=args.iou,
                           imgsz=args.imgsz, device=args.device, verbose=False)[0]
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [np.floor(xyxy[0]), np.floor(xyxy[1]),
                                           np.ceil(xyxy[2]), np.ceil(xyxy[3])])
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cls_id = int(box.cls[0].item())
            detections.append((x1, y1, x2, y2, cls_id,
                               str(model.names[cls_id]), float(box.conf[0].item())))

    # Stable order: top to bottom, then left to right.
    detections.sort(key=lambda d: (d[1], d[0]))
    annotated = image.copy()
    photo_dir = out_dir / f"photo{photo_index}"
    photo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (x1, y1, x2, y2, cls_id, name, confidence) in enumerate(detections, 1):
        obj_id = f"P{photo_index}_{number:02d}"
        roi = image[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)
        edges = cv2.Canny(blurred, args.canny_low, args.canny_high)
        save_image(photo_dir / f"{obj_id}_ROI.png", roi)
        save_image(photo_dir / f"{obj_id}_Canny.png", edges)
        # Side-by-side comparison with original ROI and Canny result.
        side_by_side = np.hstack((roi, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        save_image(photo_dir / f"{obj_id}_compare.png", side_by_side)

        cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        label = f"{obj_id} {name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        top = max(0, y1 - th - baseline - 8)
        left = min(x1, max(0, width - tw - 8))
        cv2.rectangle(annotated, (left, top), (min(width - 1, left + tw + 8),
                      min(height - 1, top + th + baseline + 8)), (0, 90, 0), -1)
        cv2.putText(annotated, label, (left + 4, top + th + 4), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        rows.append({"photo": photo_index, "object_id": obj_id, "class": name,
                     "class_id": cls_id, "confidence": round(confidence, 4),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "roi_width": x2 - x1, "roi_height": y2 - y1})

    save_image(out_dir / f"photo{photo_index}_detection.png", annotated)
    classes = sorted({row["class"] for row in rows})
    summary = {"photo": photo_index, "source": str(image_path),
               "original_width": width, "original_height": height,
               "object_count": len(rows), "distinct_class_count": len(classes),
               "classes": classes,
               "average_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4) if rows else None,
               "meets_three_class_requirement": len(classes) >= 3}
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="HW3 YOLO + per-object Canny")
    parser.add_argument("--photo1", default="photo1.jpg")
    parser.add_argument("--photo2", default="photo2.jpg")
    parser.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained detection weights")
    parser.add_argument("--out", default="HW3_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for first CUDA GPU")
    parser.add_argument("--blur", type=int, default=5, help="odd Gaussian kernel size")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    args = parser.parse_args()
    if args.blur < 1 or args.blur % 2 == 0:
        parser.error("--blur must be a positive odd integer")
    if not (0 <= args.canny_low < args.canny_high <= 255):
        parser.error("Canny thresholds must satisfy 0 <= low < high <= 255")
    if not (0 < args.conf <= 1 and 0 < args.iou <= 1):
        parser.error("conf and iou must be in (0, 1]")
    paths = [Path(args.photo1).expanduser().resolve(), Path(args.photo2).expanduser().resolve()]
    for path in paths:
        if not path.is_file():
            parser.error(f"Photo not found: {path}")
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    if len(model.names) != 80:
        raise ValueError("The model must use the 80-class COCO detection label set")
    all_rows, summaries = [], []
    for i, path in enumerate(paths, 1):
        rows, summary = analyze_photo(model, path, i, out_dir, args)
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"Photo {i}: {summary['object_count']} objects; "
              f"{summary['distinct_class_count']} classes: {', '.join(summary['classes'])}")
        if not summary["meets_three_class_requirement"]:
            print(f"WARNING: Photo {i} has fewer than 3 detected classes. Retake the photo if necessary.")

    fields = ["photo", "object_id", "class", "class_id", "confidence",
              "x1", "y1", "x2", "y2", "roi_width", "roi_height"]
    with (out_dir / "detections.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    environment = {"os": platform.platform(), "python": sys.version.split()[0],
                   "opencv": cv2.__version__, "numpy": np.__version__,
                   "ultralytics": ultralytics.__version__, "model_weights": args.model,
                   "device": args.device, "confidence_threshold": args.conf,
                   "iou_threshold": args.iou, "inference_image_size": args.imgsz,
                   "gaussian_blur_kernel": [args.blur, args.blur],
                   "canny_thresholds": [args.canny_low, args.canny_high],
                   "coordinate_format": "xyxy: (x1,y1) top-left inclusive, (x2,y2) bottom-right exclusive; original image pixels",
                   "photos": summaries}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
