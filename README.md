# next-23-MTL | Self-Sizing Elastic Neural Networks (SS-ENN v1.0)
**الشبكات العصبية المرنة ذاتية التشكيل — إطار تعلم موضعي خالٍ من الانتشار العكسي للتوليد المتسلسل والتعلم المستمر**

*A Backpropagation-Free Local Learning Framework via Spring Synapse Dynamics, Sparse Causal Attention, and Dynamic Neurogenesis*

[![Zero-Autograd AST Verified](https://img.shields.io/badge/Autograd-0%25%20(AST%20Verified)-success)](./test_suite.py)
[![Exact Match Accuracy](https://img.shields.io/badge/Exact%20Match%20(W0%20Freeze)-100.00%25-brightgreen)](./RESEARCH_PAPER_SS_ENN.md)
[![Continual Retention](https://img.shields.io/badge/Zero--Replay%20Retention-100.00%25-blue)](./benchmarks/bench_continual.py)
[![Self-Sizing Capacity Saved](https://img.shields.io/badge/Dormant%20Pruning%20Saved-56.2%25--62.5%25-orange)](./benchmarks/bench_neurogenesis.py)
[![Python 3.12 | PyTorch](https://img.shields.io/badge/Python-3.12%20%7C%20PyTorch-informational)](https://pytorch.org/)

---

## نظرة عامة | Overview

يقدم مستودع **`next-23-MTL`** أول تنفيذ مرجعي مكتمل لمعمارية **الشبكات العصبية المرنة ذاتية التشكيل (SS-ENN)**؛ وهي معمارية توليدية متسلسلة تتدرب **بدون أي انتشار عكسي للخطأ (`Zero-Backpropagation` / `Zero-Autograd`)** ودون الاحتفاظ بشجرة الحسابات (`Computational Graph`).

بدلاً من تمثيل الرابط العصبي كرقم قياسي أصم ($w$)، يُنمذج كل مشبك كـ **نابض ميكانيكي مرن ثلاثي الحالة** $\mathcal{S}_{ij} = (w_{ij}, w_{0,ij}, k_{ij})$:
1. **$w_{ij}$ (`Fast Elastic Stretch`):** التمدد اللحظي السريع الذي يلتقط أثر المشاركة الفيزيائية أثناء مرور الإشارة.
2. **$w_{0,ij}$ (`Mechanical Anchor`):** نقطة الارتكاز المستقرة التي تترسب فيها المعرفة طويلة الأمد.
3. **$k_{ij} \in [0, 1]$ (`Adaptive Stiffness`):** معامل الصلابة التكيفي الذي يحمي الذاكرة المتعلمة من النسيان الكارثي عبر قوة الإرجاع $-\gamma k(w - w_0)$.

وعند انتهاء التدريب (`production_freeze`)، تُتلف متغيرات التدريب $(w, k)$ وتُقلَّم الخلايا النائمة التي لم تُستدعَ، لتُصدَّر الشبكة كمصفوفة ارتكاز خفيفة وثابتة ($W_{\text{final}} = W_0$) جاهزة للإنتاج المباشر.

---

## المخطط المعماري | System Architecture

```text
                [Input Token Stream: w_1, w_2, ..., w_t]
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │     Sparse Causal Self-Attention     │ ◄── ALiBi Distance Bias + Theme Anchor
                │      (Top-K = 3..5, Zero Leakage)    │     Local Spring Q, K, V Projections
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │      4x Dynamic Spring Layers        │ ◄── Direct Error Broadcast B^(l) e_t
                │      Active Pool (D_init = 32)       │     (No Chain Rule / Zero Autograd)
                │    + Dormant Reserve (D_max = 128)   │ ◄── Wake-Up Trigger (+8 cells on Saturation)
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │       Production Freeze Export       │ ──► Discard (w, k) & Prune Dormant Pool
                │          W_final = W_0 Only          │     Fast Static Inference Matrix
                └──────────────────────────────────────┘
```

---

## الحصيلة المعملية للتجارب الست | Empirical Benchmark Suite (v1.0)

تخضع جميع ملفات المستودع لفحص الشجرة النحوية المجردة (`AST Zero-Autograd Verification`) للتثبت برمجياً من الانعدام التام لأي استدعاء لـ `backward()` أو `grad`:

| # | الاختبار المعملي (`Benchmark`) | السكريبت التنفيذي | التحدي وحجم البيانات | الأبعاد النشطة النهائية ($D_{\text{active}}$) | توفير السعة (`Dormant Saved`) | الدقة بعد التجميد ($W_0$ فقط) | النتيجة الفيزيائية البارزة |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **النواة الأساسية (Core MVP)** | [`train_eval.py`](./train_eval.py) | 12 جملة ثلاثية (36 كلمة + `<BOS>`) | `[64, 64, 64, 64]` | — | **100.00%** (12/12) | تقارب كامل للخطأ التوليدي `0.0000` عند الحقبة 25 |
| **2** | **ضغط السعة (Capacity Stress)** | [`benchmarks/bench_capacity.py`](./benchmarks/bench_capacity.py) | $12 \to 25 \to 50$ جملة (145 مفردة) | `[64, 64, 64, 64]` | — | **100.00%** (50/50) | عدالة توزيع المسارات (`Entropy = 0.8392`، استغلال `92.6%`) |
| **3** | **التعلم المستمر (Continual Learning)** | [`benchmarks/bench_continual.py`](./benchmarks/bench_continual.py) | مرحلتان متتاليتان ($12 + 12$ جملة) `Zero-Replay` | `[64, 64, 64, 64]` | — | **100.00%** (24/24) | احتفاظ بالذاكرة `100%`؛ الصلابة $k$ حمت الثقة عند `96.9%` مقابل `87.4%` بدونها |
| **4** | **الانتباه السببي المتفرق (Sparse Attention)** | [`benchmarks/bench_sparse_attention.py`](./benchmarks/bench_sparse_attention.py) | 12 جملة طويلة (6 كلمات بعبارات وسطى متطابقة) | `[64, 64, 64, 64]` | — | **100.00%** (12/12) | فك اشتباك بعيد المدى بتركيز انتباه حتى `82.0%` على المبتدأ (`Leakage = 0.0`) |
| **5** | **التوليد العصبي الديناميكي (Neurogenesis)** | [`benchmarks/bench_neurogenesis.py`](./benchmarks/bench_neurogenesis.py) | 100 جملة (`tier_4_ultra` - 284 مفردة) | توسع ذاتي من `32` إلى **`[56, 56, 56, 56]`** | **56.2%** (`288/512` خلية) | **100.00%** (100/100) | انهار الثابت ($D=32$) عند `52%` ($\bar{k}=0.1065$) ونجح المتوسع بإجهاد مريح ($\bar{k}=0.0092$) |
| **6** | **السلاسل الحرة المتصلة (Continuous Rollout)** | [`benchmarks/bench_continuous_stream.py`](./benchmarks/bench_continuous_stream.py) | 8 فقرات متصلة (136 كلمة، 128 انتقالاً، 48 كلمة ربط) | توسع ذاتي من `32` إلى **`[48, 48, 48, 48]`** | **62.5%** (`320/512` خلية) | **100.00%** (8/8 فقرات) | توليد ذاتي حر لـ 17 كلمة متصلة من بادئة كلمتين فقط وفك اشتباك `48/48` كلمة ربط |

---

## هيكلية المستودع | Repository Structure

```text
next-23-MTL/
├── engine/
│   ├── __init__.py                 # Engine module exports
│   ├── spring_layer.py             # Core 3-parameter spring synapse (w, w0, k) & Top-K WTA
│   ├── dfa_router.py               # Fixed Random Direct Feedback Alignment (B^(l) matrices)
│   ├── causal_context.py           # Autoregressive causal context processor
│   ├── sparse_attention.py         # Zero-autograd Sparse Causal Self-Attention (Top-K = 3..5)
│   └── dynamic_spring_layer.py     # Dormant reserve pool, frustration trigger & pruning
├── benchmarks/
│   ├── corpus_data.py              # Tiered Arabic corpora (12, 25, 50, 100 sentences & 8 paragraphs)
│   ├── bench_capacity.py           # Benchmark 1: Capacity & saturation stress test
│   ├── bench_continual.py          # Benchmark 2: Two-phase zero-replay continual learning
│   ├── bench_sparse_attention.py   # Benchmark 3: Long-range 6-word dependency disambiguation
│   ├── bench_neurogenesis.py       # Phase 1 Maturity: Self-sizing topology (32 -> 56 cells)
│   └── bench_continuous_stream.py  # Phase 2 Maturity: Continuous paragraphs & 17-word free rollout
├── model.py                        # Base SpringGenerativeNetwork (MVP)
├── model_v2.py                     # SpringGenerativeNetworkV2 (with SparseCausalAttention)
├── model_dynamic.py                # DynamicSpringGenerativeNetwork (with Dormant Reserve)
├── model_continuous.py             # ContinuousSpringNetwork (Hybrid ALiBi + Attention + Dynamic Pool)
├── train_eval.py                   # MVP training loop & W0 production freeze verification
├── test_suite.py                   # Strict unit & AST Zero-Autograd verification suite
├── SPEC_SPRING_MVP.md              # Core mathematical & architectural specification
├── PAPER_SS_ENN.md                 # Full technical research paper (Arabic/English)
├── RESEARCH_PAPER_SS_ENN.md        # Comprehensive equations, empirical tables & reproducibility guide
└── LIMITATIONS_AND_V2_ROADMAP.md   # Critical scaling analysis & SS-ENN v2.0 execution roadmap
```

---

## التشغيل السريع وإعادة الإنتاج | Quick Start & Reproducibility

### 1. استنساخ المستودع (Clone Repository)
```bash
git clone https://github.com/naoufelbvb2-arch/next-23-MTL.git
cd next-23-MTL
```

### 2. تشغيل حزمة التحقق الصارم وفحص AST (Run Verification Suite)
```bash
python test_suite.py
python train_eval.py
```

### 3. تشغيل الاختبارات المعملية الخمسة (Run All Benchmarks)
```bash
# 1. Capacity & Saturation Benchmark (12 -> 25 -> 50 sentences)
python benchmarks/bench_capacity.py

# 2. Continual Learning Zero-Replay Benchmark (Phase 1 -> Phase 2)
python benchmarks/bench_continual.py

# 3. Sparse Causal Self-Attention Long-Range Benchmark (6-word sentences)
python benchmarks/bench_sparse_attention.py

# 4. Dynamic Neurogenesis & Self-Sizing Topology (100 sentences)
python benchmarks/bench_neurogenesis.py

# 5. Continuous Token Streams & 17-Word Free Autoregressive Rollout (8 paragraphs)
python benchmarks/bench_continuous_stream.py
```

---

## التوثيق العلمي وخارطة الطريق | Research Documentation & v2.0 Roadmap

- **[التوثيق البحثي الشامل ودليل إعادة الإنتاج (`RESEARCH_PAPER_SS_ENN.md`)](./RESEARCH_PAPER_SS_ENN.md):** الاشتقاق النظري لقوانين النابض الثلاثة، معادلات التوجيه المباشر (`DFA`)، جداول النتائج الرقمية الكاملة، ودليل إعادة الإنتاج.
- **[الورقة البحثية التقنية (`PAPER_SS_ENN.md`)](./PAPER_SS_ENN.md):** النص الأكاديمي الكامل لمعمارية `SS-ENN v1.0` وتحليل الظواهر الفيزيائية المرصودة.
- **[وثيقة الحدود المعمارية وخارطة طريق الإصدار الثاني (`LIMITATIONS_AND_V2_ROADMAP.md`)](./LIMITATIONS_AND_V2_ROADMAP.md):** مراجعة نقدية شاملة للسبع عقبات القياسية نحو نماذج `1B` (التعميم التركيبي، تخفيف إشارة `DFA` في القواميس الضخمة، التضمين الموضعي، طور النوم وإعادة تدوير الخلايا، والتفرق الكتلي المتوافق مع `GPU`).
- **[مواصفات النواة الأساسية (`SPEC_SPRING_MVP.md`)](./SPEC_SPRING_MVP.md):** وثيقة المواصفات الأولية والقيود الصارمة للنظام.

---

## المؤلف | Author

**Naoufel Hamdani** ([@naoufelbvb2-arch](https://github.com/naoufelbvb2-arch))
