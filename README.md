# Sport-App
# ⚡ AERO-KINETIX AI: Multi-Sport Conditioning, Biomechanical Telemetry & Scouting Lab

**AERO-KINETIX AI** is a full-stack sports science analytics and AI coaching platform built in pure Python using **Streamlit**, **Plotly WebGL**, and **Groq / OpenRouter LLMs**. Designed for high-performance athletic monitoring, it bridges physiological load modeling, interactive 3D biomechanical visualization, and real-time generative AI prescription.

---

## 🧠 Core Architecture & Sports Science Models

### 1. Physiological Readiness Index ($R_{\text{index}}$)
The daily readiness algorithm computes a normalized composite score $[0, 100]$ from sleep restoration ($S$), neuromuscular soreness inversion ($M$), subjective CNS drive ($E$), and an Acute:Chronic Workload Ratio ($\text{ACWR}$) fatigue penalty ($\Phi$):

$$S_{\text{norm}} = \min\left(100, \left(\frac{\text{Hours Slept}}{8.5}\right) \times 100\right)$$

$$M_{\text{norm}} = \left(\frac{10 - \text{Soreness}}{9}\right) \times 100, \quad E_{\text{norm}} = \left(\frac{\text{Energy} - 1}{9}\right) \times 100$$

$$R_{\text{raw}} = 0.40 \cdot S_{\text{norm}} + 0.35 \cdot M_{\text{norm}} + 0.25 \cdot E_{\text{norm}}$$

$$R_{\text{index}} = \text{clip}\left(R_{\text{raw}} - \Phi(\text{ACWR}), 5, 100\right)$$

### 2. Acute:Chronic Workload Ratio ($\text{ACWR}$)
Internal training load is quantified via Session Rating of Perceived Exertion ($\text{sRPE} = \text{RPE}_{1-10} \times \text{Duration}_{\text{min}}$) in Arbitrary Units ($\text{AU}$):
- **Acute Load ($A_7$):** 7-day rolling mean daily workload.
- **Chronic Load ($C_{28}$):** 28-day rolling baseline capacity.
- **Ratio ($\text{ACWR} = A_7 / C_{28}$):**
  - `< 0.80`: **Under-training / Detraining Zone**
  - `0.80 – 1.30`: **Optimal Adaptation ("Sweet Spot")**
  - `1.31 – 1.50`: **Functional Overreaching (Caution Zone)**
  - `> 1.50`: **Non-Functional Overreaching / High Injury Risk Spike**

### 3. Interactive 3D WebGL Biomechanical Avatar
Built natively with `plotly.graph_objects` (`go.Scatter3d` skeletal segments/joints + `go.Mesh3d` core volume + force-plate telemetry grid). Features:
- Full 360° mouse orbital rotation, pan, and scroll-wheel zoom.
- Dynamic accent color synchronization with the active pro athlete's jersey theme.
- Real-time anatomical heatmap highlighting localized muscle soreness selected in the sidebar.

### 4. Dual-Mode AI Tactical Coach (Groq / OpenRouter + Heuristic Fallback)
- Connects to **Groq** (`llama-3.3-70b-versatile` or `llama3-8b-8192`) or **OpenRouter** to synthesize a concise, 3-sentence physiological prescription justifying volume scaling and targeted myofascial recovery.
- Includes a deterministic **Sports Science Heuristic Fallback Engine** so the application runs offline without crashing during academic grading if no API key is supplied.

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python `3.9+` installed on your system.

### Step 1: Clone or Create Project Directory
Place `requirements.txt`, `README.md`, and `app.py` inside a single folder:

```bash
mkdir aero-kinetix-ai
cd aero-kinetix-ai
