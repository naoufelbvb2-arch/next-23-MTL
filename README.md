# next-23-MTL | Self-Sizing Elastic Neural Networks (SS-ENN v1.0 & v2.0 Milestones 2.1–2.3)
**الشبكات العصبية المرنة ذاتية التشكيل — إطار تعلم موضعي خالٍ من الانتشار العكسي للتوليد المتسلسل والتعلم المستمر**

*A Backpropagation-Free Local Learning Framework via Spring Synapse Dynamics, Dual-Horizon Sparse Causal Attention, Dynamic Neurogenesis, and Dual-Stage Sleep Consolidation*

[![Zero-Autograd AST Verified](https://img.shields.io/badge/Autograd-0%25%20(25%20Files%20AST%20Verified)-success)](./tests/test_suite.py)
[![Milestone 2.3 Zero-Shot Recombination](https://img.shields.io/badge/Milestone%202.3-90.27%25%20Hybrid%20%7C%20100%25%20Binding-blue)](./benchmarks/bench_v2_compositional_gen.py)
[![Exact Match Accuracy](https://img.shields.io/badge/Exact%20Match%20(W0%20Freeze)-100.00%25-brightgreen)](./docs/RESEARCH_PAPER_SS_ENN.md)
[![Scaled CPU Validation](https://img.shields.io/badge/1.47M%20English%20Val%20Acc-93.66%25%20(PPL%201.35)-blueviolet)](./docs/V2_ENGINEERING_LOG_AND_FIXES.md)
[![4-Phase Sleep Continual Retention](https://img.shields.io/badge/4--Phase%20Sleep%20Recall-97.22%25%20(54.7%25%20Saved)-teal)](./benchmarks/bench_v2_sleep_continual.py)
[![Entity Consistency](https://img.shields.io/badge/Cross--Sentence%20Entity%20Retention-100.00%25%20(5%2F5)-blue)](./benchmarks/bench_scale_english_cpu.py)
[![Python 3.12 | PyTorch](https://img.shields.io/badge/Python-3.12%20%7C%20PyTorch-informational)](./requirements.txt)

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

## الحصيلة المعملية للتجارب التسع | Empirical Benchmark Suite (v1.0 & v2.0 Milestones 2.1–2.3)

تخضع جميع ملفات المستودع (25 ملفاً برمجياً) لفحص الشجرة النحوية المجردة (`AST Zero-Autograd Verification`) للتثبت برمجياً من الانعدام التام لأي استدعاء لـ `backward()` أو `grad`:

| # | الاختبار المعملي (`Benchmark`) | السكريبت التنفيذي | التحدي وحجم البيانات | الأبعاد النشطة النهائية ($D_{\text{active}}$) | توفير السعة / الذاكرة | الدقة بعد التجميد ($W_0$ فقط) | النتيجة الفيزيائية البارزة |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **النواة الأساسية (Core MVP)** | [`train_eval.py`](./train_eval.py) | 12 جملة ثلاثية (36 كلمة + `<BOS>`) | `[64, 64, 64, 64]` | — | **100.00%** (12/12) | تقارب كامل للخطأ التوليدي `0.0000` عند الحقبة 25 |
| **2** | **ضغط السعة (Capacity Stress)** | [`benchmarks/bench_capacity.py`](./benchmarks/bench_capacity.py) | $12 \to 25 \to 50$ جملة (145 مفردة) | `[64, 64, 64, 64]` | — | **100.00%** (50/50) | عدالة توزيع المسارات (`Entropy = 0.8392`، استغلال `92.6%`) |
| **3** | **التعلم المستمر (Continual Learning)** | [`benchmarks/bench_continual.py`](./benchmarks/bench_continual.py) | مرحلتان متتاليتان ($12 + 12$ جملة) `Zero-Replay` | `[64, 64, 64, 64]` | — | **100.00%** (24/24) | احتفاظ بالذاكرة `100%`؛ الصلابة $k$ حمت الثقة عند `96.9%` مقابل `87.4%` بدونها |
| **4** | **الانتباه السببي المتفرق (Sparse Attention)** | [`benchmarks/bench_sparse_attention.py`](./benchmarks/bench_sparse_attention.py) | 12 جملة طويلة (6 كلمات بعبارات وسطى متطابقة) | `[64, 64, 64, 64]` | — | **100.00%** (12/12) | فك اشتباك بعيد المدى بتركيز انتباه حتى `82.0%` على المبتدأ (`Leakage = 0.0`) |
| **5** | **التوليد العصبي الديناميكي (Neurogenesis)** | [`benchmarks/bench_neurogenesis.py`](./benchmarks/bench_neurogenesis.py) | 100 جملة (`tier_4_ultra` - 284 مفردة) | توسع ذاتي من `32` إلى **`[56, 56, 56, 56]`** | **56.2%** (`288/512` خلية) | **100.00%** (100/100) | انهار الثابت ($D=32$) عند `52%` ($\bar{k}=0.1065$) ونجح المتوسع بإجهاد مريح ($\bar{k}=0.0092$) |
| **6** | **السلاسل الحرة المتصلة (Continuous Rollout)** | [`benchmarks/bench_continuous_stream.py`](./benchmarks/bench_continuous_stream.py) | 8 فقرات متصلة (136 كلمة، 128 انتقالاً، 48 كلمة ربط) | توسع ذاتي من `32` إلى **`[48, 48, 48, 48]`** | **62.5%** (`320/512` خلية) | **100.00%** (8/8 فقرات) | توليد ذاتي حر لـ 17 كلمة متصلة من بادئة كلمتين فقط وفك اشتباك `48/48` كلمة ربط |
| **7** | **المحرك المليوني وفيزياء الطور 2.1 (`v2.0 CPU Scale-Up`)** | [`benchmarks/bench_scale_english_cpu.py`](./benchmarks/bench_scale_english_cpu.py) | 394 قصة إنجليزية (`25K+` رمز، `V=2,048`، `90% Train / 10% Unseen Val`) | توسع ذاتي من `64` إلى **`[192, 192, 192, 192]`** | **77.4% RAM Saved** (`25.13 MB -> 5.69 MB`, `1.47M` معامل) | **`93.84%` Train / `93.66%` Val** (`PPL 1.35`, `100%` ثبات الكيانات) | سرعة **`10,523 tok/s`** على `CPU` (الذروة `13,010 tok/s`)، صلابة $\bar{k}=0.1387$، وثبات الكيانات `5/5` عبر الجمل |
| **8** | **طور النوم المزدوج وإعادة التدوير (`v2.0 Milestone 2.2`)** | [`benchmarks/bench_v2_sleep_continual.py`](./benchmarks/bench_v2_sleep_continual.py) | 4 مراحل سردية متتالية (`384` قصة، `Zero-Replay`، `V=2,048`) | نمو منضبط من `32` إلى **`[116, 116, 116, 116]`** (مقابل `256` الساذج) | **54.7% Cells Saved** (`464/1024`, **`4.86 MB`**, `192` خلية معاد تدويرها) | **`97.22%` Mean Recall (`PPL 1.14`)** vs `83.45%` Naive (**`+13.77%`**) | **`144`** خلية أُعيد إحياؤها موضعياً، **`100%` (`5/5`)** ثبات الكيانات، و**`0.00%`** تكرار متجاور (`"the the"`) أو حلقات عبارة |
| **9** | **الصلابة الهرمية، تطبيع الطاقة الموضعي، والتعميم التركيبي الصفري (`v2.0 Milestone 2.3`)** | [`benchmarks/bench_v2_compositional_gen.py`](./benchmarks/bench_v2_compositional_gen.py) | تدريب قطري (`64` قصة) + تقييم صفري (`W_0`) على `24` توليفة هجينة غير مرئية `(L_j, S_i, A_k)` | طيف Top-K هرمي `[24, 24, 20, 16]` وصلابة متدرجة `k_bar = [0.0799..0.1676]` | تثبيت الطاقة الموضعي `RMS = 1.2000` عبر `L1..L4` (مقابل تضخم `1.7607` المسطح) | **`90.27%` Hybrid Full (`PPL 1.870`)**، **`97.01%` Hybrid Cont**، **`100.00%` (`48/48`)** Subject Binding (`0.00%` Reversion) | **`100.00%` (`6/6`)** توليد حر (`+30` رمزاً) بـ `0.00%` تكرار متجاور وحلقات عبارة |

---

## هيكلية المستودع | Repository Structure

```text
next-23-MTL/
├── engine/                           # النواة الفيزيائية والرياضية (الطبقات والتوجيه والمُجزئ)
│   ├── __init__.py
│   ├── spring_layer.py               # Core 3-parameter spring synapse (w, w0, k) & Top-K WTA
│   ├── dynamic_spring_layer.py       # Dormant reserve pool, frustration trigger & pruning
│   ├── sparse_attention.py           # Zero-autograd Sparse Causal Self-Attention (Top-K = 3..5)
│   ├── dfa_router.py                 # Fixed & Adaptive Direct Feedback Alignment (B^(l) matrices)
│   ├── causal_context.py             # Autoregressive causal context processor
│   └── tokenizer.py                  # Hybrid English Word + BPE Subword Tokenizer (V=2048..4096)
├── models/                           # حزمة النماذج المعزولة (v1.0 & v2.0 Scaled)
│   ├── __init__.py
│   ├── model_mvp.py                  # Base SpringGenerativeNetwork (MVP)
│   ├── model_v2_attn.py              # SpringGenerativeNetworkV2 (with SparseCausalAttention)
│   ├── model_dynamic.py              # DynamicSpringGenerativeNetwork (with Dormant Reserve)
│   ├── model_continuous.py           # ContinuousSpringNetwork (Hybrid ALiBi + Attention + Dynamic Pool)
│   ├── model_scaled.py               # ScaledSpringNetwork (1.5M-5M Vectorized + Milestones 2.1–2.3)
├── benchmarks/                       # حزمة التجارب المعملية والبيانات
│   ├── __init__.py
│   ├── corpus_data.py                # Tiered Arabic corpora (12, 25, 50, 100 sentences & 8 paragraphs)
│   ├── english_dataset.py            # TinyStories-style connected English narrative dataset (90/10 split)
│   ├── bench_capacity.py             # Benchmark 1: Capacity & saturation stress test
│   ├── bench_continual.py            # Benchmark 2: Two-phase zero-replay continual learning
│   ├── bench_sparse_attention.py     # Benchmark 3: Long-range 6-word dependency disambiguation
│   ├── bench_neurogenesis.py         # Phase 1 Maturity: Self-sizing topology (32 -> 56 cells)
│   ├── bench_continuous_stream.py    # Phase 2 Maturity: Continuous paragraphs & 17-word free rollout
│   ├── bench_scale_english_cpu.py    # v2.0 Milestone 2.1: Million-param CPU telemetry & 6-table audit
│   ├── bench_v2_sleep_continual.py   # v2.0 Milestone 2.2: 4-phase sleep consolidation, recycling & refractory gate
│   └── bench_v2_compositional_gen.py # v2.0 Milestone 2.3: Hierarchical stiffness, Active RMSNorm & zero-shot recombination
├── tests/                            # حزمة التحقق الصارم وفحص AST
│   ├── __init__.py
│   └── test_suite.py                 # Strict unit & AST Zero-Autograd verification suite (25 files, 7 tests)
├── docs/                             # الأوراق البحثية وسجل الإصلاحات وخارطة الطريق
│   ├── RESEARCH_PAPER_SS_ENN.md      # Full technical paper, equations, empirical tables & reproducibility
│   ├── V2_ENGINEERING_LOG_AND_FIXES.md # Detailed v2.0 Milestones 2.1–2.3 physics upgrades & audit
│   ├── LIMITATIONS_AND_V2_ROADMAP.md # Critical scaling analysis & SS-ENN v2.0 execution roadmap
│   └── SPEC_SPRING_MVP.md            # Core mathematical & architectural specification
├── .gitignore
├── requirements.txt                  # اعتماديات المشروع (torch>=2.0.0, pytest>=7.0.0)
├── train_eval.py                     # نقطة التشغيل السريع للنواة (MVP)
└── README.md                         # الواجهة الرسمية للمستودع
```

---

## التشغيل السريع وإعادة الإنتاج | Quick Start & Reproducibility

### 1. استنساخ المستودع وتثبيت المتطلبات (Clone & Install)
```bash
git clone https://github.com/naoufelbvb2-arch/next-23-MTL.git
cd next-23-MTL
pip install -r requirements.txt
```

### 2. تشغيل حزمة التحقق الصارم وفحص AST (Run Verification Suite)
```bash
python tests/test_suite.py
python train_eval.py
```

### 3. تشغيل الاختبارات المعملية الثمانية (Run All v1.0 & v2.0 Benchmarks)
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

# 6. Million-Parameter CPU Scale-Up & v2.0 Core Physics Benchmark (Milestone 2.1)
python benchmarks/bench_scale_english_cpu.py

# 7. 4-Phase Zero-Replay Continual Learning, Dual-Stage Sleep & Synaptic Recycling (Milestone 2.2)
python benchmarks/bench_v2_sleep_continual.py

# 8. Hierarchical Stiffness, Local Active RMSNorm & Zero-Shot Compositional Generalization (Milestone 2.3)
python benchmarks/bench_v2_compositional_gen.py
```

---

## التوثيق العلمي وخارطة الطريق | Research Documentation & v2.0 Roadmap

- **[الورقة البحثية المتكاملة (`docs/RESEARCH_PAPER_SS_ENN.md`)](./docs/RESEARCH_PAPER_SS_ENN.md):** الاشتقاق النظري لقوانين النابض الثلاثة، معادلات التوجيه المباشر (`DFA`)، التجارب التسع الكاملة، وتحليل ظاهرة التشكيل الذاتي وترشيح الضوضاء عبر نقطة الارتكاز (`Anchor Denoising Effect`).
- **[سجل الإصلاحات والترقيات الفيزيائية للإصدار الثاني (`docs/V2_ENGINEERING_LOG_AND_FIXES.md`)](./docs/V2_ENGINEERING_LOG_AND_FIXES.md):** التوثيق الرياضي والبرمجي لقوانين **`v2.0 Milestones 2.1–2.3`** (معالجة انجراف الكيانات، تفتت الصلابة الدفعية، طور النوم المزدوج، إعادة تدوير الخلايا، الدرع التكعيبي، بوابة الجموح الانتقالية، الطيف الهرمي للصلابة، تطبيع الطاقة الموضعي، والتعميم التركيبي الصفري).
- **[وثيقة الحدود المعمارية وخارطة طريق الإصدار الثاني (`docs/LIMITATIONS_AND_V2_ROADMAP.md`)](./docs/LIMITATIONS_AND_V2_ROADMAP.md):** مراجعة نقدية شاملة للسبع عقبات القياسية نحو نماذج `1B` ومصفوفة التقدم التنفيذية لأطوار `v2.0`.
- **[مواصفات النواة الأساسية (`docs/SPEC_SPRING_MVP.md`)](./docs/SPEC_SPRING_MVP.md):** وثيقة المواصفات الأولية والقيود الصارمة للنظام.

---

## المؤلف | Author

**Naoufel Hamdani** ([@naoufelbvb2-arch](https://github.com/naoufelbvb2-arch))
