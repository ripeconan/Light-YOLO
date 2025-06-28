**Light-YOLO: Repository for “Light-YOLO: A Lightweight Framework for Multi‑Scale Contraband Detection in X-ray Security Images via Channel‑Decoupled Feature Learning”**

> **Note:** This manuscript is currently under major revision.

---

## Repository Structure

```
├── cfg/
│   ├── models/               # model definitions (YOLOv8 variants)
│   │   ├── light_yolon.yaml  # N variant (width=0.25×, depth=0.25×)
│   │   ├── light_yolos.yaml  # S variant (width=0.50×, depth=0.33×)
│   │   ├── light_yolom.yaml  # M variant (width=0.75×, depth=0.67×)
│   │   └── light_yolol.yaml  # L variant (width=1.00×, depth=1.00×)
│   └── datasets/            # dataset settings
│       ├── opixray.yaml      # OPIXray dataset
│       └── hixray.yaml       # HiXray dataset
├── nn/
│   └── modules/             # core model components
│       ├── starnet_modules.py  # SE‑StarNet backbone blocks
│       ├── csc_modules.py      # Channel‑Separated Convolution blocks
│       └── head_modules.py     # LSCCD detection head
├── scripts/
│   └── convert_hixray_opixray_to_yolo.py  # raw → YOLO format converter
└── README.md                  # this file
```

---

## Installation

```bash
pip install ultralytics
```
Then copy this repo into the corresponding position of the ultralytics program.

---

## Usage

1. **Data Preparation**

   ```bash
   python scripts/convert_hixray_opixray_to_yolo.py 
   ```

2. **Training & Validation**

   ```bash
   # train on OPIXray
   yolo task=detect mode=train \
     model=cfg/modes/light_yolos.yaml \
     data=cfg/datasets/opixray.yaml \
     epochs=200 imgsz=640 batch=16 \
     project=runs/exp name=light_yolos_opixray

   # validate mAP & FPS
   yolo task=detect mode=val \
     model=runs/exp/light_yolos_opixray/weights/best.pt \
     data=cfg/datasets/opixray.yaml \
     batch=16
   ```

3. **Inference**

   ```bash
   yolo task=detect mode=predict \
     model=runs/exp/light_yolos_opixray/weights/best.pt \
     source=data/OPIXray/yolo/images/test save=True
   ```

---

## Configuration Files

- **Model YAMLs:** located in `cfg/modes/`, define architecture and scaling for n/s/m/l variants.
- **Dataset YAMLs:** located in `cfg/datasets/`, specify `train`, `val`, `test` paths and class info.

---

## Core Modules

- **SE-StarNet** (`nn/modules/starnet_modules.py`): efficient backbone with SE attention.
- **CSC** (`nn/modules/csc_modules.py`): channel‑decoupled conv with cross‑reconstruction.
- **LSCCD** (`nn/modules/head_modules.py`): lightweight detection head using CSC.

---

## Citation

This repository supports the major revision of the manuscript. For citation, please use:

> Ran Wang, Yang Zhou, Guanghuan Hu, Xianghua Xu. “Light-YOLO: A Lightweight Framework for Multi‑Scale Contraband Detection in X‑ray Security Images via Channel‑Decoupled Feature Learning.” Manuscript under major revision.

---

## License

Licensed under MIT. See `LICENSE` for details.

---

## Contributing

Issues and pull requests are welcome for bug fixes or enhancements.
