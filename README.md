# 🚀 HMEAYC - 幼兒律動行為 AI 鑑定與社交網絡分析系統
> **2026 大專競賽參賽作品** | 基於 YOLOv8 與 MediaPipe 的專業鑑定方案

![Mobile View Preview](https://github.com/user-attachments/assets/your-screenshot-link-here)

## 🌟 核心亮點 (Project Highlights)
- **鑑定級流暢播放**：優化 OpenCV 幀緩存技術，實現毫秒級精準行為比對。
- **3 步驟極簡流程**：從影片上傳、AI 辨識到社交圖譜，一氣呵成。
- **2026 行動優先設計**：完美適配手機與平板，支援跨裝置隨時查閱分析報表。
- **專業數據導出**：支援 Excel 報表與互動圖譜下載，滿足專業觀察需求。

## 🛠️ 技術架構 (Technology Stack)
- **AI 辨識核心**：YOLOv8 (物件偵測) + MediaPipe (骨架分析)
- **後端邏輯**：Python / Streamlit
- **數據存儲**：SQLite (本機備份) + Hugging Face DB (雲端同步)
- **互動視覺化**：Plotly / NetworkX (社交圖譜)

## 📱 功能預覽
1. **設定與上傳**：支援多種活動模式（模仿、創作）與偵測精度設定。
2. **分析與報表**：YouTube 等級的播放器與專業觀察表單同步作業。
3. **社交網絡圖**：自動辨識班級中的社交核心 (Social Hubs) 與互動頻率。

## 🚀 快速開始 (Quick Start)
如果您是在本機執行：
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔐 隱私與安全性
本系統嚴格遵守「即時毀棄」原則，所有影片影像僅於運算過程中處理，不存儲任何幼兒臉部原始資料，僅保留骨架特徵數據。

---
© 2026 HMEAYC AI Team. All rights reserved.
