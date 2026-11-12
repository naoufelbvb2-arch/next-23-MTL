# سجل الإصلاحات الهندسية والترقيات الفيزيائية للإصدار الثاني (SS-ENN v2.0 — Milestone 2.1)
**Engineering Log, Bottleneck Diagnostics, and Permanent v2.0 Core Physics Upgrades on the Million-Parameter CPU Engine**

---

## 1. مدخل المقياس المليوني والمعالجة الدفعية المصفوفية (`Scale-Up to 1.47M Preset & Vectorized Mini-Batching`)

عند الانتقال بمعمارية **الشبكات العصبية المرنة ذاتية التشكيل (`SS-ENN`)** من المدونات التجريبية المصغرة (`v1.0` بحجم $V \le 284$ ومعالجة تسلسلية فردية `batch_size = 1`) إلى النطاق المليوني الواقعي على المعالج المحلي (`CPU Real-World Scale-Up`)، تم بناء ثلاث وحدات معيارية معزولة بالكامل عن نواة `v1.0`:

1. **مُجزئ النصوص الإنجليزية الهجين ([`engine/tokenizer.py`](../engine/tokenizer.py)):**
   - الفئة `EnglishSubwordTokenizer` التي تفصل الكلمات الإنجليزية وعلامات الترقيم وتبني قاموساً قياسياً موسعاً ($V = 2,048$ في إعداد `1.5M` و $V = 4,096$ في إعداد `5M`) عبر دمج الكلمات كاملة التردد مع أزواج المقاطع (`BPE Subword Merges`) والرموز التأسيسية (`<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`).
2. **مدونة السرد الإنجليزي المتصل ([`benchmarks/english_dataset.py`](../benchmarks/english_dataset.py)):**
   - الدالة `build_english_tinystories_corpus` التي تولد مدونة سردية إنجليزية مترابطة بأسلوب `TinyStories` تتجاوز `25,000+` رمز عبر `394` فقرة قصصية متعددة الجمل، مقسمة بصرامة إلى:
     - **`Train Split (90%)`:** `355` فقرة (`12,780` رمزاً فعلياً في نوافذ التدريب).
     - **`Validation Split (10% Held-Out Compositional)`:** `39` فقرة (`1,404` رموز) مؤلفة من توليفات سردية لم يرها النموذج حرفياً أثناء التدريب لقياس التعميم الحقيقي (`Validation Perplexity & Next-Token Accuracy`).
3. **المحرك المليوني المصفوفي ([`models/model_scaled.py`](../models/model_scaled.py)):**
   - الفئة `ScaledSpringNetwork` التي تدعم إعدادَي `1.5M` (`V=2048, L=4, D_init=64, D_max=256, K=16`) و `5M` (`V=4096, L=6, D_init=128, D_max=512, K=32`).
   - بدلاً من حلقات `for` التسلسلية الفردية، تُنفذ جميع عمليات الانتباه السببي والنافذة الموضعية (`ALiBi`) وطبقات النوابض الديناميكية كعمليات مصفوفية ثلاثية الأبعاد `[Batch_Size=32, Seq_Len=36, D_max=256]` مدمجة في `[B * T, D_max]`، مما رفع سرعة التدريب على المعالج المحلي (`CPU`) إلى **`8,500 – 12,171` رمز/ثانية** وسرعة الاستدلال الدفعي بعد التجميد إلى **`27,088` رمز/ثانية**.

---

## 2. التشخيص الهندسي للثغرتين المرصودتين في النسخة المليونية الأولى (`Baseline v1-Scaled Diagnostics`)

رغم نجاح النسخة المليونية الأولى في تحقيق سرعة تتجاوز `10,000` رمز/ثانية ودقة تحقق `92.09%` (`Val PPL = 1.38`)، كشف التدقيق الفيزيائي الدقيق عن ظاهرتين هيكليتين تمنعان ثبات التوليد المستمر طويل المدى:

### 2.1 الثغرة الأولى: انجراف الكيان عبر الجمل (`Cross-Sentence Entity Drift`)
- **الظاهرة المرصودة:**
  أثناء التوليد الحر المستمر (`Free-Running Rollout` باستخدام $W_0$ فقط)، عندما بدأ النموذج بالبادئة رقم `#04` (`"The young rabbit"`):
  > *"The young rabbit hopped along the soft path one sunny afternoon in the small village**. afterward the little bird** flew across the clear sky..."*
  انجرفت الجملة الثانية بعد جسر الربط السردي `". afterward"` من شخصية الأرنب (`"young rabbit"`) إلى شخصية الطائر (`"little bird"`).
- **التفسير الفيزيائي الدقيق للثغرة:**
  1. **احتكار النوابض بواسطة الكلمات عالية التكرار (`Hub Word Monopoly`):** تتكرر الكلمات الوظيفية والروابط السردية (`"the"`, `"."`, `"afterward"`, `"then"`) في كل فقرة تقريباً، فتستحوذ على حصة الأسد من تحديثات النوابض $\Delta w$ وتطغى على بصمة الكيانات الدلالية النادرة نسبياً (`"rabbit"`, `"bird"`, `"squirrel"`).
  2. **مسح ذاكرة الفاعل بواسطة الجار والمجرور الختامي (`Prepositional Tail Overwrite`):** في الجمل التي تبدأ بالفاعل (`"The young rabbit hopped along the soft path one sunny afternoon in the small village . afterward the"`), تفصل بين الفاعل (`"young rabbit"` في الموضعين 2 و3) وبين كلمة `"afterward"` (في الموضع 17) اثنتا عشرة كلمة تنتهي بعبارة المكان المشتركة `"in the small village"`. وفي غياب مكثف سياقي محكوم بالأهمية، تهيمن عبارة المكان المشتركة على التمثيل السياقي لحظة عبور `". afterward"`، فينسى النموذج هوية الفاعل الأصلي.

### 2.2 الثغرة الثانية: تفتت الصلابة في التدريب الدفعي (`Mini-Batch Stiffness Dilution`)
- **الظاهرة المرصودة:**
  في النسخة المليونية الأولى، توقف متوسط الصلابة النهائي عند $\bar{k} = 0.0001$ (شبه صفري) بعد 12 حقبة كاملة.
- **التفسير الفيزيائي الدقيق للثغرة:**
  في المعالجة الدفعية المصفوفية بحجم $N = B \times T = 32 \times 36 = 1,152$ رمزاً، يُفعّل قناع التنافسية `Top-K` ($K = 16$ من أصل $D_{\text{active}}$) نسبة ضئيلة فقط من الخلايا لكل رمز. وعند قسمة إشارة الدفع المجمعة $\sum_{n=1}^{N} \delta_{n,j} a_{n,i}$ على العدد الكلي $N = 1,152$ بدلاً من عدد مرات فوز الخلية الفعلي $N_{\text{wins}}(j)$، تتضاءل خطوة التمدد اللحظي $|w_{ij} - w_{0,ij}|$ بمقدار $10\times$ إلى $16\times$، فينهار تحديث الصلابة $\Delta k_{ij} = \beta_k |w_{ij} - w_{0,ij}| (1 - k_{ij})$ إلى الصفر العملي (`0.0001`).

---

## 3. القوانين الفيزيائية الأربعة للإصلاح الدائم (`Permanent v2.0 Core Physics Laws`)

تمت معالجة الظاهرتين جذرياً عبر أربعة قوانين فيزيائية عامة ومستقلة عن أي بيانات خاصة، ومطبقة بالكامل داخل [`models/model_scaled.py`](../models/model_scaled.py) و [`benchmarks/bench_scale_english_cpu.py`](../benchmarks/bench_scale_english_cpu.py):

### 3.1 القانون الأول: الترجيح اللوغاريتمي المقيّد لأهمية الرموز (`Law 1: Clipped Log-Salience Weighting`)
- **الموقع البرمجي:** الدالة [`ScaledSpringNetwork.calibrate_salience_from_tokens`](../models/model_scaled.py#L294-L369) (الأسطر `294–369`) وتطبيقها في [`train_minibatch`](../models/model_scaled.py#L609-L616) (الأسطر `609–616`).
- **الصياغة الرياضية:**
  يُحسب لكل رمز $v \in \{1, \dots, V\}$ مؤشر معلومات مركب $I(v)$ يجمع بين الندرة اللوغاريتمية للرمز والوثيقة (`Inverse Log TF + IDF`) وبين **معامل التواتر الموضوعي داخل الفقرة (`Intra-Document Burstiness`)** الذي يميز الكيانات المحورية التي تتكرر عبر جمل الفقرة الواحدة عن الكلمات الوظيفية أو الظروف العابرة:
  $$\text{IDF}(v) = \log\!\left(\frac{N_{\text{docs}} + 1}{\text{DF}(v) + 1}\right), \qquad \text{Burst}(v) = \frac{\sum_{d=1}^{N_{\text{docs}}} \mathbb{I}\!\left(\text{count}(v, d) \ge 2\right)}{\text{DF}(v) + 1} \cdot \mathbb{I}\!\left(\text{DF}(v) < 0.30\,N_{\text{docs}}\right)$$
  $$I(v) = 0.55 \log\!\left(\frac{N_{\text{total}} + 1}{\text{count}(v) + 1}\right) + 0.45\,\text{IDF}(v) + 1.85\,\text{Burst}(v)$$
  ثم يُطبَّع $I(v)$ خطياً ويُقيَّد داخل النطاق الفيزيائي الآمن $[0.70, 1.80]$:
  $$s(v) = \text{clip}\!\left(0.70 + 1.10 \cdot \frac{I(v) - I_{\min}}{I_{\max} - I_{\min}}, \; 0.70, \; 1.80\right)$$
  ويُرجَّح متجه الخطأ اللحظي في كل خطوة زمنية $t$ بمتوسط أهمية الرمز الهدف والرمز المدخل:
  $$\mathbf{e}_t^{\text{salient}} = \text{clip}\!\left(0.65\,s(y_t) + 0.35\,s(x_t), \; 0.70, \; 1.80\right) \cdot \left(\mathbf{y}_t - \hat{\mathbf{y}}_t\right)$$
- **الأثر الفيزيائي المرصود:**
  انخفضت أوزان الكلمات الوظيفية والروابط إلى `{'the': 0.70, '.': 0.85, 'afterward': 1.001, 'then': 1.008}`، بينما ارتفعت أوزان الكيانات المحورية إلى `{'rabbit': 1.518, 'bird': 1.531, 'squirrel': 1.545, 'kitten': 1.560}`.

---

### 3.2 القانون الثاني: مكثف السياق المستمر المحكوم بالأهمية والانتباه ثنائي الأفق (`Law 2: Salience-Gated EMA Capacitor + Dual-Horizon Sparse Attention`)
- **الموقع البرمجي:**
  - الفئة [`DualHorizonCausalAttention`](../models/model_scaled.py#L65-L141) (الأسطر `65–141`).
  - الدالة [`ScaledSpringNetwork._build_vectorized_context`](../models/model_scaled.py#L442-L483) (الأسطر `442–483`).
- **الصياغة الرياضية:**
  1. **مكثف السياق المستمر المحكوم ببوابة الأهمية (`Salience-Gated EMA Context Capacitor`):**
     يحتفظ كل تسلسل بمتجه مكثف سياقي سببي $\mathbf{c}_t \in \mathbb{R}^{D_{\max}}$ يتحدث عند كل رمز $x_t$ وفق:
     $$\mathbf{c}_t = \left(1 - \alpha(x_t)\right)\mathbf{c}_{t-1} + \alpha(x_t)\,\text{Embed}(x_t), \qquad \mathbf{c}_0 = \mathbf{0}$$
     حيث تُحكم بوابة الشحن $\alpha(v)$ بعتبة الأهمية الدلالية للرمز:
     $$\alpha(v) = \begin{cases} 0.38 + 0.12\,\tilde{s}(v), & \text{if } s(v) \ge 1.45 \;\text{(Thematic Entity Latch)} \\[4pt] 0.004\,\tilde{s}(v)^2, & \text{if } s(v) < 1.45 \;\text{(Syntactic / Hub Hold)} \end{cases}, \qquad \tilde{s}(v) = \frac{s(v) - 0.70}{1.10}$$
     عندما يمر الفاعل المحوري في الجملة الأولى (مثل `"young rabbit"` أو `"little bird"` حيث $s(v) > 1.50$)، يشحن المكثف بقوة ($\alpha \approx 0.48$). وعندما تمر عبارات الظرف والمكان وجسور الربط (`"in the small village . afterward the"` حيث $s(v) < 1.30$)، تنغلق بوابة الكتابة ($\alpha \le 0.004$) فيحتفظ المكثف بأكثر من **93%** من شحنة الفاعل الأصلي حتى لحظة توليد فاعل الجملة الثانية!
  2. **الانتباه السببي المتفرق ثنائي الأفق (`Dual-Horizon Top-K Causal Attention`):**
     مع الحفاظ الصارم على القناع السببي $M_{t,\tau} = -\infty$ لكل $\tau > t$ (`Future Leakage = 0.0`)، تُقسَّم مقاعد الانتباه $K_{\text{attn}} = 8$ عند كل موضع استعلام $t$ إلى:
     - **$K_{\text{local}} = K_{\text{attn}} - 2 = 6$ مقاعد محلية:** تختار أعلى 6 مفاتيح في الدرجة السياقية المباشرة.
     - **$K_{\text{anchor}} = 2$ مقعدي ارتكاز دلالي (`Salient Anchor Slots`):** يختاران أعلى رمزين في دالة الارتكاز المحكومة بالأهمية عبر البادئة السببية $\tau \in \{0, \dots, t\}$:
       $$S_{\text{anchor}}(t, \tau) = 4.0\,s(x_\tau) + 0.15\,\text{Score}(t, \tau) \quad (\tau \le t)$$

---

### 3.3 القانون الثالث: معايرة التحديث والتصلب بعدد مرات الفوز الفعلي (`Law 3: Winner-Count Normalized Spring & Maturity-Gated Stiffening`)
- **الموقع البرمجي:** الدالة [`ScaledSpringNetwork._update_dynamic_layer_winner_normalized`](../models/model_scaled.py#L522-L580) (الأسطر `522–580`) وجدول النضج في [`train_minibatch`](../models/model_scaled.py#L624-L629) (الأسطر `624–629`).
- **الصياغة الرياضية:**
  لكل خلية نشطة $j \in \{1, \dots, D_{\text{active}}\}$، يُحسب عدد مرات فوزها الفعلي في قناع `Top-K` عبر الدفعة:
  $$N_{\text{wins}}(j) = \max\!\left(\sum_{n=1}^{B \cdot T} m_{n,j}^{(l)}, \; 1\right), \qquad r_{\text{sparse}} = \frac{D_{\text{active}}^{(l)}}{K}$$
  وتُعاير قوة الدفع ومعامل التصلب التدريجي $\beta_k^{\text{eff}}(e, j)$ وفق:
  $$F_{ji}^{(l)} = \frac{\sum_{n=1}^{B \cdot T} \delta_{n,j}^{(l)}\,a_{n,i}^{(l-1)}}{0.40\,N_{\text{valid}} + 0.60\,N_{\text{wins}}(j)\,r_{\text{sparse}}}$$
  $$\Delta k_{ji}^{(l)} = \beta_k \cdot G_{\text{maturity}}(e) \cdot \text{clip}\!\left(\frac{N_{\text{valid}}}{N_{\text{wins}}(j)\,r_{\text{sparse}}}, \; 1.0, \; 8.0\right) \cdot \left|w_{ji}^{(l)} - w_{0,ji}^{(l)}\right| \left(1 - k_{ji}^{(l)}\right) \cdot \mathbb{I}\!\left(N_{\text{wins}}(j) > 0\right)$$
  حيث يبدأ عامل النضج $G_{\text{maturity}}(e) = 2.0$ في الحقبتين `1–2` للسماح بالتوليد العصبي السلس، ثم يرتفع تدريجياً $G_{\text{maturity}}(e) = \min(280.0, \; 32.0(e - 1))$ لتثبيت الروابط الناضجة.
- **الأثر الفيزيائي المرصود:**
  ارتفع متوسط الصلابة النهائي $\bar{k}$ من `0.0001` إلى **`0.1389`** (موزعاً بانتظام عبر الطبقات الأربع: `[0.1279, 0.1375, 0.1389, 0.1511]`)، محققاً شرط النطاق الفيزيائي المستهدف **`[0.12, 0.35]` (`PASS`)** مع ارتفاع دقة التحقق إلى **`93.66%`**.

---

### 3.4 القانون الرابع: توجيه الخطأ المتكيف محلياً (`Law 4: Row-Normalized Hebbian Feedback Alignment`)
- **الموقع البرمجي:** [`train_minibatch`](../models/model_scaled.py#L649-L656) (الأسطر `649–656`) والتدمير الكامل عند التجميد في [`production_freeze`](../models/model_scaled.py#L724-L726) (الأسطر `724–726`).
- **الصياغة الرياضية:**
  تتكيف مصفوفة بث الخطأ المباشر $B^{(l)} \in \mathbb{R}^{D_{\text{active}} \times V}$ موضعياً عبر الترابط الهبي بين التفعيل المحلي $\mathbf{a}^{(l)}$ ومتجه الخطأ المرجح بالأهمية $\mathbf{e}^{\text{salient}}$، متبوعاً بتطبيع صفّي (`Row-Wise L2 Normalization`) يمنع الانفجار العددي:
  $$\tilde{B}^{(l)} = B^{(l)} + \frac{\eta_B}{N_{\text{valid}}} \left(\mathbf{A}^{(l)}\right)^T \mathbf{E}^{\text{salient}}, \qquad B_{j,:}^{(l)} \leftarrow \sqrt{\frac{V}{D_{\max}}} \cdot \frac{\tilde{B}_{j,:}^{(l)}}{\left\|\tilde{B}_{j,:}^{(l)}\right\|_2 + \epsilon}$$
  وعند استدعاء `production_freeze()`، تُحذف مصفوفات $B^{(l)}$ بالكامل (`B_l = None`) جنباً إلى جنب مع $w$ و $k$.

---

## 4. جدول المقارنة الشامل (قبل الإصلاح vs. بعد الإصلاح) والجداول الستة الكاملة

### 4.1 مقارنة الأداء الفيزيائي والتوليدي قبل وبعد ترقية `v2.0 Milestone 2.1`

| المؤشر الهندسي / الفيزيائي | النسخة المليونية الأولى (`Pre-Fix Baseline`) | بعد الترقية الفيزيائية (`v2.0 Milestone 2.1`) | التحسن والأثر الهندسي |
| :--- | :---: | :---: | :--- |
| **دقة التدريب بعد التجميد (`Frozen W0 Train Acc`)** | `92.70%` | **`93.88%`** | **+1.18%** تحسن في استقرار الارتكاز |
| **حيرة التدريب بعد التجميد (`Frozen W0 Train PPL`)** | `1.36` | **`1.33`** | انخفاض في الحيرة والخطأ المتبقي |
| **دقة التحقق غير المرئي (`Frozen W0 Unseen Val Acc`)** | `92.09%` | **`93.66%`** | **+1.57%** تحسن في التعميم التركيبي |
| **حيرة التحقق غير المرئي (`Frozen W0 Unseen Val PPL`)** | `1.38` | **`1.35`** | تعميم أقوى على التوليفات الجديدة |
| **متوسط الصلابة النهائي ($\bar{k}$)** | `0.0001` *(OUT OF BAND)* | **`0.1389`** *(`PASS` ضمن `[0.12, 0.35]`)* | علاج تفتت الصلابة الدفعي بـ **1,389x** |
| **توزيع الصلابة عبر الطبقات الأربع** | `[0.0001, 0.0001, 0.0001, 0.0001]` | **`[0.1279, 0.1375, 0.1389, 0.1511]`** | تدرج فيزيائي متماسك عبر العمق |
| **معدل ثبات الكيانات عبر الجمل (`Entity Consistency`)** | `80.00% (4/5)` *(انجراف الأرنب إلى الطائر)* | **`100.00% (5/5 PASS — NO DRIFT)`** | القضاء التام على انجراف الكيانات بعد `". afterward"` و `". then"` |
| **فحص انعدام الانتشار العكسي (`AST Zero-Autograd`)** | `PASS` | **`PASS (23 Python files verified)`** | صفر استدعاء لـ `backward()` أو `.grad` |

### 4.2 مقارنة التوليد الحر للبادئة `#04 ("The young rabbit")` قبل وبعد الإصلاح
- **قبل الإصلاح (`Pre-Fix — Entity Drift after ". afterward"`):**
  ```text
  Seed Prompt : "The young rabbit"
  Generated   : The young rabbit hopped along the soft path one sunny afternoon in the small village. afterward the little bird flew across the clear sky and built a safe nest in
  Status      : DRIFT (Sentence-1: "young rabbit" -> Bridge: ". afterward" -> Sentence-2: "little bird")
  ```
- **بعد الإصلاح (`Post-Fix v2.0 — 100% Entity Retention`):**
  ```text
  Seed Prompt : "The young rabbit"
  Generated   : The young rabbit hopped along the soft path in the bright morning near the quiet river. afterward the young rabbit found a fresh carrot in the grass and shared the food with
  Status      : PASS / NO DRIFT (Sentence-1: "young rabbit" -> Bridge: ". afterward" -> Sentence-2: "young rabbit")
  ```

---

### 4.3 الجداول الستة الكاملة لمخرجات [`benchmarks/bench_scale_english_cpu.py`](../benchmarks/bench_scale_english_cpu.py)

#### الجدول (1): تفصيل المعاملات واستهلاك الذاكرة (`TABLE 1: Parameter & RAM Footprint Breakdown`)
| المرحلة / المؤشر الهندسي | إعداد `1.5M Preset` (المنفّذ فعلياً) | إعداد `5M Preset` (المرجعي) |
| :--- | :---: | :---: |
| حجم المفردات ($V$) والبعد الأقصى ($D_{\max}$) | $V = 2,048,\; D_{\max} = 256$ | $V = 4,096,\; D_{\max} = 512$ |
| إجمالي المعاملات الكامنة أثناء التدريب (`Latent Max at D_max`) | `6,302,720` معامل | `30,955,520` معامل |
| المعاملات النشطة الابتدائية (`Initial Active at D_init`) | `2,812,928` معامل | `12,376,064` معامل |
| المعاملات النشطة المستيقظة قبل التجميد (`Awakened Active`) | `5,041,152` معامل (`D=[192]*4`) | ديناميكي حسب الطلب |
| **معاملات الإنتاج النهائية بعد التجميد والتقليم (`Frozen W0 Only`)** | **`1,469,440` معامل (~1.47M)** | **`4,315,136` معامل (~4.32M)** |
| بصمة ذاكرة التدريب الحية (`w + w0 + k + DFA B`) | `25.13 MB` | `124.26 MB` |
| **بصمة ذاكرة الإنتاج بعد التجميد (`W0 Only`)** | **`5.69 MB`** | **`16.46 MB`** |
| **نسبة التوفير الفعلي في الذاكرة (`RAM Saved %`)** | **`77.4% saved`** | **`86.1% saved`** |

#### الجدول (2): سرعة المعالج المحلي وزمن الاستجابة (`TABLE 2: CPU Throughput & Latency Telemetry`)
| مؤشر الأداء على المعالج المحلي (CPU) | القيمة المسجلة |
| :--- | :---: |
| **متوسط سرعة التدريب المصفوفي (`Vectorized Mini-Batch Training`)** | **`8,497 Tokens / sec`** (الذروة: **`12,171 Tokens / sec`**) |
| متوسط زمن الحقبة الواحدة (`Wall-Clock Time per Epoch`) | **`1.50 sec / epoch`** (`18.05s` لإجمالي 12 حقبة) |
| **سرعة التقييم الدفعي بعد التجميد (`Frozen W0 Batch Evaluation`)** | **`27,088 Tokens / sec`** |
| **سرعة التوليد التسلسلي الحر (`Autoregressive Generation Speed`)** | **`225.7 Tokens / sec`** (**`4.43 ms / token`**) |

#### الجدول (3): منحنى التقارب وفجوة التعميم (`TABLE 3: Train vs. Validation Generalization Audit`)
| `Epoch` | `Train Loss` | `Train PPL` | `Train Acc` | `Val Loss` | `Val PPL` | `Val Acc` | `Mean k` | `Active Dims` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01` | `5.8355` | `342.23` | `20.56%` | `4.2121` | `67.50` | `37.18%` | `0.0002` | `[96, 96, 96, 96]` |
| `02` | `2.9927` | `19.94` | `52.06%` | `1.8444` | `6.32` | `67.66%` | `0.0010` | `[96, 96, 96, 96]` |
| `04` | `0.8972` | `2.45` | `87.80%` | `0.9370` | `2.55` | `84.26%` | `0.0378` | `[128, 128, 128, 128]` |
| `06` | `0.5692` | `1.77` | `90.88%` | `0.5641` | `1.76` | `90.46%` | `0.0957` | `[128, 128, 128, 128]` |
| `08` | `0.4828` | `1.62` | `91.35%` | `0.4566` | `1.58` | `91.24%` | `0.1042` | `[160, 160, 160, 160]` |
| `10` | `0.3255` | `1.38` | `93.45%` | `0.4393` | `1.55` | `90.74%` | `0.1052` | `[192, 192, 192, 192]` |
| `12` | `0.3336` | `1.40` | `92.74%` | `0.3493` | `1.42` | `93.95%` | `0.1389` | `[192, 192, 192, 192]` |
| **`FROZEN W0`** | **`0.2864`** | **`1.33`** | **`93.88%`** | **`0.3010`** | **`1.35`** | **`93.66%`** | **`0.1389`** | **`[192, 192, 192, 192]`** |

#### الجدول (4): التشخيص الفيزيائي للنوابض والتوليد العصبي (`TABLE 4: Physical Spring & Neurogenesis Diagnostics`)
- **الأبعاد النشطة الابتدائية (`D_init`):** `[64, 64, 64, 64]`
- **الأبعاد النشطة النهائية بعد 4 مراحل إيقاظ (`Final Awakened Dims`):** `[192, 192, 192, 192]` من أصل `[256, 256, 256, 256]`
- **الخلايا المشاركة فعلياً في `Top-K`:** `[192, 192, 192, 192]` (`100%` استغلال للخلايا المستيقظة)
- **الخلايا النائمة الموفرة والمقلمة عند التجميد:** `256 / 1024` (**`25.0% capacity saved`**)
- **إنتروبيا مشاركة الخلايا (`Neuron Participation Entropy`):** **`0.9625`** (`[0.9652, 0.9629, 0.9623, 0.9596]`)
- **متوسط الصلابة النهائي (`Final Mean Stiffness k_bar`):** **`0.1389`** — **`PASS`** ضمن النطاق `[0.12, 0.35]` (`[0.1279, 0.1375, 0.1389, 0.1511]`)

#### الجدول (5): عينات التوليد الإنجليزي الحر المستمر بعد التجميد (`TABLE 5: Free-Running English Generation Samples — W0 Only`)
| `#` | `Seed Prompt` | `Generated Paragraph (+30 Tokens using Frozen W0 Only)` |
| :-: | :--- | :--- |
| `01` | `"Once upon a time"` | `Once upon a time in the green valley, the brave puppy ran happily around the yard. then the brave puppy chased a colorful butterfly near a flowers the brave puppy chased a` |
| `02` | `"The little bird"` | `The little bird sang a sweet song in the small village. afterward the little bird flew across the clear sky and built a safe nest in the branches. suddenly the curious` |
| `03` | `"In the bright morning"` | `In the bright morning near the quiet river, the the curious boy built a small wooden boat by the shore. then the curious boy sailed the toy boat across the calm pond` |
| `04` | `"The young rabbit"` | `The young rabbit hopped along the soft path in the bright morning near the quiet river. afterward the young rabbit found a fresh carrot in the grass and shared the food with` |
| `05` | `"Under the tall oak"` | `Under the tall oak tree in the ancient forest, the clever fox walked quietly through the woods. then the clever fox watched the silver fish in the stream and learned a wise` |

#### الجدول (6): تدقيق ثبات الكيانات عبر الجمل (`TABLE 6: Cross-Sentence Entity Consistency Audit`)
| `Prompt #` | `Seed Prompt` | `Sentence-1 Entity` | `Hub Bridge` | `Sentence-2 Entity` | `Status` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `#01` | `"Once upon a time"` | `brave puppy` | `. then` | `brave puppy` | **`PASS (NO DRIFT)`** |
| `#02` | `"The little bird"` | `little bird` | `. afterward` | `little bird` | **`PASS (NO DRIFT)`** |
| `#03` | `"In the bright morning"` | `curious boy` | `. then` | `curious boy` | **`PASS (NO DRIFT)`** |
| `#04` | `"The young rabbit"` | `young rabbit` | `. afterward` | `young rabbit` | **`PASS (NO DRIFT)`** |
| `#05` | `"Under the tall oak"` | `clever fox` | `. then` | `clever fox` | **`PASS (NO DRIFT)`** |
| **Total** | **Consistency Rate ($W_0$ Only)** | — | — | **5 / 5** | **`100.00% PASS`** |

---

## 5. طريقة إعادة الإنتاج والتحقق (`Reproducibility`)
```bash
# تشغيل اختبار المحرك المليوني والترقية الفيزيائية v2.0 على المعالج المحلي
python benchmarks/bench_scale_english_cpu.py

# تشغيل حزمة الاختبارات الصارمة وفحص AST عبر 23 ملفاً برمجياً
python tests/test_suite.py
```
