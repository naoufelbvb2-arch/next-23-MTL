# سجل الإصلاحات الهندسية والترقيات الفيزيائية للإصدار الثاني (SS-ENN v2.0 — Milestones 2.1, 2.2, 2.3 & 2.4)
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
   - بدلاً من حلقات `for` التسلسلية الفردية، تُنفذ جميع عمليات الانتباه السببي والنافذة الموضعية (`ALiBi`) وطبقات النوابض الديناميكية كعمليات مصفوفية ثلاثية الأبعاد `[Batch_Size=32, Seq_Len=36, D_max=256]` مدمجة في `[B * T, D_max]`، مما رفع متوسط سرعة التدريب على المعالج المحلي (`CPU`) إلى **`10,523` رمز/ثانية** (الذروة **`13,010` رمز/ثانية**) وسرعة الاستدلال الدفعي بعد التجميد إلى **`38,660` رمز/ثانية**.

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
- **الموقع البرمجي:** الدالة [`ScaledSpringNetwork.calibrate_salience_from_tokens`](../models/model_scaled.py#L331-L415) (الأسطر `331–415`) وتطبيقها في [`train_minibatch`](../models/model_scaled.py#L812-L821) (الأسطر `812–821`).
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
  - الدالة [`ScaledSpringNetwork._build_vectorized_context`](../models/model_scaled.py#L504-L544) (الأسطر `504–544`).
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
- **الموقع البرمجي:** الدالة [`ScaledSpringNetwork._update_dynamic_layer_winner_normalized`](../models/model_scaled.py#L707-L767) (الأسطر `707–767`) وجدول النضج في [`train_minibatch`](../models/model_scaled.py#L829-L833) (الأسطر `829–833`).
- **الصياغة الرياضية:**
  لكل خلية نشطة $j \in \{1, \dots, D_{\text{active}}\}$، يُحسب عدد مرات فوزها الفعلي في قناع `Top-K` عبر الدفعة:
  $$N_{\text{wins}}(j) = \max\!\left(\sum_{n=1}^{B \cdot T} m_{n,j}^{(l)}, \; 1\right), \qquad r_{\text{sparse}} = \frac{D_{\text{active}}^{(l)}}{K}$$
  وتُعاير قوة الدفع ومعامل التصلب التدريجي $\beta_k^{\text{eff}}(e, j)$ وفق:
  $$F_{ji}^{(l)} = \frac{\sum_{n=1}^{B \cdot T} \delta_{n,j}^{(l)}\,a_{n,i}^{(l-1)}}{0.40\,N_{\text{valid}} + 0.60\,N_{\text{wins}}(j)\,r_{\text{sparse}}}$$
  $$\Delta k_{ji}^{(l)} = \beta_k \cdot G_{\text{maturity}}(e) \cdot \text{clip}\!\left(\frac{N_{\text{valid}}}{N_{\text{wins}}(j)\,r_{\text{sparse}}}, \; 1.0, \; 8.0\right) \cdot \left|w_{ji}^{(l)} - w_{0,ji}^{(l)}\right| \left(1 - k_{ji}^{(l)}\right) \cdot \mathbb{I}\!\left(N_{\text{wins}}(j) > 0\right)$$
  حيث يبدأ عامل النضج $G_{\text{maturity}}(e) = 2.0$ في الحقبتين `1–2` للسماح بالتوليد العصبي السلس، ثم يرتفع تدريجياً $G_{\text{maturity}}(e) = \min(280.0, \; 32.0(e - 1))$ لتثبيت الروابط الناضجة.
- **الأثر الفيزيائي المرصود:**
  ارتفع متوسط الصلابة النهائي $\bar{k}$ من `0.0001` إلى **`0.1387`** (موزعاً بانتظام عبر الطبقات الأربع: `[0.1277, 0.1374, 0.1388, 0.1510]`)، محققاً شرط النطاق الفيزيائي المستهدف **`[0.12, 0.35]` (`PASS`)** مع ارتفاع دقة التحقق إلى **`93.66%`**.

---

### 3.4 القانون الرابع: توجيه الخطأ المتكيف محلياً (`Law 4: Row-Normalized Hebbian Feedback Alignment`)
- **الموقع البرمجي:** [`train_minibatch`](../models/model_scaled.py#L870-L877) (الأسطر `870–877`) والتدمير الكامل عند التجميد في [`production_freeze`](../models/model_scaled.py#L1413-L1414) (الأسطر `1413–1414`).
- **الصياغة الرياضية:**
  تتكيف مصفوفة بث الخطأ المباشر $B^{(l)} \in \mathbb{R}^{D_{\text{active}} \times V}$ موضعياً عبر الترابط الهبي بين التفعيل المحلي $\mathbf{a}^{(l)}$ ومتجه الخطأ المرجح بالأهمية $\mathbf{e}^{\text{salient}}$، متبوعاً بتطبيع صفّي (`Row-Wise L2 Normalization`) يمنع الانفجار العددي:
  $$\tilde{B}^{(l)} = B^{(l)} + \frac{\eta_B}{N_{\text{valid}}} \left(\mathbf{A}^{(l)}\right)^T \mathbf{E}^{\text{salient}}, \qquad B_{j,:}^{(l)} \leftarrow \sqrt{\frac{V}{D_{\max}}} \cdot \frac{\tilde{B}_{j,:}^{(l)}}{\left\|\tilde{B}_{j,:}^{(l)}\right\|_2 + \epsilon}$$
  وعند استدعاء `production_freeze()`، تُحذف مصفوفات $B^{(l)}$ بالكامل (`B_l = None`) جنباً إلى جنب مع $w$ و $k$.

---

## 4. جدول المقارنة الشامل (قبل الإصلاح vs. بعد الإصلاح) والجداول الستة الكاملة

### 4.1 مقارنة الأداء الفيزيائي والتوليدي قبل وبعد ترقية `v2.0 Milestone 2.1`

| المؤشر الهندسي / الفيزيائي | النسخة المليونية الأولى (`Pre-Fix Baseline`) | بعد الترقية الفيزيائية (`v2.0 Milestone 2.1`) | التحسن والأثر الهندسي |
| :--- | :---: | :---: | :--- |
| **دقة التدريب بعد التجميد (`Frozen W0 Train Acc`)** | `92.70%` | **`93.84%`** | **+1.14%** تحسن في استقرار الارتكاز |
| **حيرة التدريب بعد التجميد (`Frozen W0 Train PPL`)** | `1.36` | **`1.33`** | انخفاض في الحيرة والخطأ المتبقي |
| **دقة التحقق غير المرئي (`Frozen W0 Unseen Val Acc`)** | `92.09%` | **`93.66%`** | **+1.57%** تحسن في التعميم التركيبي |
| **حيرة التحقق غير المرئي (`Frozen W0 Unseen Val PPL`)** | `1.38` | **`1.35`** | تعميم أقوى على التوليفات الجديدة |
| **متوسط الصلابة النهائي ($\bar{k}$)** | `0.0001` *(OUT OF BAND)* | **`0.1387`** *(`PASS` ضمن `[0.12, 0.35]`)* | علاج تفتت الصلابة الدفعي بـ **1,387x** |
| **توزيع الصلابة عبر الطبقات الأربع** | `[0.0001, 0.0001, 0.0001, 0.0001]` | **`[0.1277, 0.1374, 0.1388, 0.1510]`** | تدرج فيزيائي متماسك عبر العمق |
| **معدل ثبات الكيانات عبر الجمل (`Entity Consistency`)** | `80.00% (4/5)` *(انجراف الأرنب إلى الطائر)* | **`100.00% (5/5 PASS — NO DRIFT)`** | القضاء التام على انجراف الكيانات بعد `". afterward"` و `". then"` |
| **فحص انعدام الانتشار العكسي (`AST Zero-Autograd`)** | `PASS` | **`PASS (24 Python files verified)`** | صفر استدعاء لـ `backward()` أو `.grad` |

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
| **متوسط سرعة التدريب المصفوفي (`Vectorized Mini-Batch Training`)** | **`10,523 Tokens / sec`** (الذروة: **`13,010 Tokens / sec`**) |
| متوسط زمن الحقبة الواحدة (`Wall-Clock Time per Epoch`) | **`1.21 sec / epoch`** (`14.57s` لإجمالي 12 حقبة) |
| **سرعة التقييم الدفعي بعد التجميد (`Frozen W0 Batch Evaluation`)** | **`38,660 Tokens / sec`** |
| **سرعة التوليد التسلسلي الحر (`Autoregressive Generation Speed`)** | **`228.5 Tokens / sec`** (**`4.38 ms / token`**) |

#### الجدول (3): منحنى التقارب وفجوة التعميم (`TABLE 3: Train vs. Validation Generalization Audit`)
| `Epoch` | `Train Loss` | `Train PPL` | `Train Acc` | `Val Loss` | `Val PPL` | `Val Acc` | `Mean k` | `Active Dims` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01` | `5.8355` | `342.23` | `20.56%` | `4.2121` | `67.50` | `37.18%` | `0.0002` | `[96, 96, 96, 96]` |
| `02` | `2.9927` | `19.94` | `52.06%` | `1.8444` | `6.32` | `67.66%` | `0.0010` | `[96, 96, 96, 96]` |
| `04` | `0.8972` | `2.45` | `87.80%` | `0.9370` | `2.55` | `84.26%` | `0.0378` | `[128, 128, 128, 128]` |
| `06` | `0.5692` | `1.77` | `90.88%` | `0.5641` | `1.76` | `90.46%` | `0.0957` | `[128, 128, 128, 128]` |
| `08` | `0.4828` | `1.62` | `91.35%` | `0.4566` | `1.58` | `91.31%` | `0.1042` | `[160, 160, 160, 160]` |
| `10` | `0.3243` | `1.38` | `93.48%` | `0.4489` | `1.57` | `91.03%` | `0.1052` | `[192, 192, 192, 192]` |
| `12` | `0.3088` | `1.36` | `93.06%` | `0.3501` | `1.42` | `93.87%` | `0.1387` | `[192, 192, 192, 192]` |
| **`FROZEN W0`** | **`0.2838`** | **`1.33`** | **`93.84%`** | **`0.2966`** | **`1.35`** | **`93.66%`** | **`0.1387`** | **`[192, 192, 192, 192]`** |

#### الجدول (4): التشخيص الفيزيائي للنوابض والتوليد العصبي (`TABLE 4: Physical Spring & Neurogenesis Diagnostics`)
- **الأبعاد النشطة الابتدائية (`D_init`):** `[64, 64, 64, 64]`
- **الأبعاد النشطة النهائية بعد 4 مراحل إيقاظ (`Final Awakened Dims`):** `[192, 192, 192, 192]` من أصل `[256, 256, 256, 256]`
- **الخلايا المشاركة فعلياً في `Top-K`:** `[192, 192, 192, 192]` (`100%` استغلال للخلايا المستيقظة)
- **الخلايا النائمة الموفرة والمقلمة عند التجميد:** `256 / 1024` (**`25.0% capacity saved`**)
- **إنتروبيا مشاركة الخلايا (`Neuron Participation Entropy`):** **`0.9625`** (`[0.9653, 0.9629, 0.9623, 0.9595]`)
- **متوسط الصلابة النهائي (`Final Mean Stiffness k_bar`):** **`0.1387`** — **`PASS`** ضمن النطاق `[0.12, 0.35]` (`[0.1277, 0.1374, 0.1388, 0.1510]`)

#### الجدول (5): عينات التوليد الإنجليزي الحر المستمر بعد التجميد (`TABLE 5: Free-Running English Generation Samples — W0 Only`)
| `#` | `Seed Prompt` | `Generated Paragraph (+30 Tokens using Frozen W0 Only)` |
| :-: | :--- | :--- |
| `01` | `"Once upon a time"` | `Once upon a time in the green valley, the brave puppy ran happily around the yard. then the brave puppy chased a colorful butterfly near a flowers the guarded the wooden gate` |
| `02` | `"The little bird"` | `The little bird sang a sweet song, the little bird flew across the clear sky and built a safe nest in the branches and the gentle deer drank cool water from the` |
| `03` | `"In the bright morning"` | `In the bright morning near the quiet river, the little bird sang a sweet song. then the little bird flew across the clear sky and built a safe nest in the branches` |
| `04` | `"The young rabbit"` | `The young rabbit hopped along the soft path in the bright morning near the quiet river. afterward the young rabbit found a fresh carrot in the grass and shared the food with` |
| `05` | `"Under the tall oak"` | `Under the tall oak tree in the ancient forest, the busy squirrel gathered shiny acorns for the winter. then the busy squirrel climbed up the tallest pine tree and stored the nuts` |

#### الجدول (6): تدقيق ثبات الكيانات عبر الجمل (`TABLE 6: Cross-Sentence Entity Consistency Audit`)
| `Prompt #` | `Seed Prompt` | `Sentence-1 Entity` | `Hub Bridge` | `Sentence-2 Entity` | `Status` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `#01` | `"Once upon a time"` | `brave puppy` | `. then` | `brave puppy` | **`PASS (NO DRIFT)`** |
| `#02` | `"The little bird"` | `little bird` | `. (direct)` | `little bird` | **`PASS (NO DRIFT)`** |
| `#03` | `"In the bright morning"` | `little bird` | `. then` | `little bird` | **`PASS (NO DRIFT)`** |
| `#04` | `"The young rabbit"` | `young rabbit` | `. afterward` | `young rabbit` | **`PASS (NO DRIFT)`** |
| `#05` | `"Under the tall oak"` | `busy squirrel` | `. then` | `busy squirrel` | **`PASS (NO DRIFT)`** |
| **Total** | **Consistency Rate ($W_0$ Only)** | — | — | **5 / 5** | **`100.00% PASS`** |

---

## 5. الطور 2.2: طور النوم المزدوج، إعادة تدوير الخلايا، قفل الدرع التكعيبي، وفترة الجموح الانتقالية (`Milestone 2.2: Dual-Stage Sleep Consolidation, Synaptic Recycling, Cubic Armor Locking & Transition Refractory Gate`)

### 5.1 الدافع الهندسي والظواهر المرصودة بعد الطور 2.1
عند الانتقال من التدريب أحادي الطور (`Single-Phase`) إلى **التدريب المتسلسل متعدد المراحل دون إعادة عرض البيانات (`4-Phase Zero-Replay Continual Learning`)** على المحرك المليوني (`models/model_scaled.py`)، برزت أربع مشكلات فيزيائية وهندسية:
1. **التضخم أحادي الاتجاه (`Monotonic Layer Inflation`):** في التدريب الساذج (`Naive Continual`)، يستيقظ كامل الحوض الكامن (`[256, 256, 256, 256]`) بحلول المرحلة الثانية دون أي آلية لدمج الخلايا المتشابهة أو تقليم الخلايا منخفضة المنفعة وإعادة تدويرها.
2. **التداخل الكارثي عبر المراحل (`Cross-Phase Interference`):** غياب قفل الدرع الميكانيكي (`Sleep Armor Locking`) والتوجيه الرنيني للفضاءات الجزئية المتخصصة (`Resonant Specialist Subspace Routing`) يؤدي إلى تدهور دقة المرحلة الأولى من `>96%` إلى `66.44%` (`PPL = 5.86`) بعد المرحلة الرابعة.
3. **التلعثم الفوري (`Immediate Adjacent Token Stutter` مثل `"the the"`):** غياب فترة جموح عصبية فورية عند الانتقال $t-1 \to t$ يسمح للكلمات الوظيفية عالية التردد بإطلاق نفسها مرتين متتاليتين.
4. **التكرار الحلقي للعبارات في ذيل التوليد الطويل (`Tail Phrase Loops`):** عند توليد 30+ رمزاً، قد تعيد الشبكة إطلاق نفس العبارة الرباعية (`4-gram`) في نهاية المقطع.

---

### 5.2 القوانين الفيزيائية الأربعة المطبقة في الطور 2.2 ([`models/model_scaled.py`](../models/model_scaled.py))

#### أولاً: طور النوم المزدوج وإعادة تدوير الخلايا (`Dual-Stage Offline Sleep: Synaptic Fusion + Low-Utility Pruning`)
يُنفَّذ عبر الدالة [`ScaledSpringNetwork.sleep_consolidate_and_recycle`](../models/model_scaled.py#L1073-L1345) عقب انتهاء كل مرحلة تدريب مستمر $p \in \{1, 2, 3, 4\}$:
1. **المرحلة الأولى — الدمج المشبكي المحافظ على الإشارة (`Stage 1: Signal-Preserving High-Cosine Synaptic Fusion`):**
   تُحسب مصفوفة تشابه جيب التمام بين متجهات الارتكاز النشطة $\mathbf{w}_{0,u}, \mathbf{w}_{0,v}$، وتُدمج الأزواج المتشابهة مع الحفاظ على معيار الإشارة $L_2$:
   $$\mathbf{w}_{0,u}^{\text{new}} = \|\mathbf{w}_{0,u}\|_2 \cdot \frac{0.70\,\mathbf{w}_{0,u} + 0.30\,\mathbf{w}_{0,v}}{\|0.70\,\mathbf{w}_{0,u} + 0.30\,\mathbf{w}_{0,v}\|_2}$$
   بينما تُصفَّر الخلية المدمجة $v$ وتُعاد إلى حوض الخلايا المعاد تدويرها (`rejuvenated_slots`).
2. **المرحلة الثانية — تقليم الخلايا المستيقظة منخفضة المنفعة (`Stage 2: Low-Utility Awakening Pruning`):**
   تُقاس المنفعة المشبكية لكل خلية بلاستيكية مستيقظة $j$:
   $$U_j = \|\mathbf{w}_{0,j}\|_2 \cdot (\bar{k}_j + 0.05) \cdot \log\!\big(2 + \text{wins}_j\big)$$
   وتُقلَّم الخلايا ذات المنفعة الأدنى وتُعاد إلى `rejuvenated_slots`.

#### ثانياً: إعادة الإحياء الموضعي والضغط المتزامن (`In-Place Dormant Rejuvenation & Synchronous Compaction`)
- **أثناء المراحل اللاحقة ([`prepare_continual_phase`](../models/model_scaled.py#L954-L1031)):** قبل زيادة العداد `d_active`، تُستدعى [`_rejuvenate_layer_slots_in_place`](../models/model_scaled.py#L931-L951) لإعادة إيقاظ الفجوات المُعاد تدويرها في `rejuvenated_slots` موضعياً (`48 خلية` في كل مرحلة من المراحل 2 و3 و4، أي `144 خلية` أُعيد إحياؤها موضعياً عبر المراحل).
- **عند التجميد الإنتاجي ([`production_freeze(prune_dormant=True)`](../models/model_scaled.py#L1348-L1436)):** تُضغط المصفوفة `W0` فعلياً لحذف أي صفوف معاد تدويرها أو نائمة مع الاحتفاظ بفهرس المحاذاة المتبقية `_compacted_indices`، مما يقلص الأبعاد النشطة النهائية إلى **`[116, 116, 116, 116]`** (`464` خلية نشطة فقط من أصل `1,024`، بتوفير **`54.7%`** من السعة).

#### ثالثاً: قفل الدرع التكعيبي والتوجيه الرنيني للفضاءات المتخصصة (`Cubic Armor Locking & Resonant Specialist Subspace Routing`)
- **الدرع التكعيبي (`Cubic Anchor Shield` — [`_update_dynamic_layer_winner_normalized`](../models/model_scaled.py#L707-L767)):** تُقفل الخلايا الناجية في نهاية كل طور نوم عند صلابة $k \in [0.88, 0.96]$ مع تطبيق معامل الحماية التكعيبي $(1 - k_{ij})^3$ على تحديثات نقطة الارتكاز $\mathbf{w}_0$:
  $$\Delta w_{0,ij} = \eta_0 \,(1 - k_{ij})^3 \,(w_{ij} - w_{0,ij}) \quad (\forall k_{ij} \ge 0.80)$$
- **التوجيه الرنيني للفضاءات الجزئية المتخصصة (`Resonant Specialist Subspace Routing` — [`_detect_resonant_phase`](../models/model_scaled.py#L629-L641)):** يحتفظ كل طور $p$ بمجموعته المنفصلة من الخلايا المتخصصة المقفلة $\mathcal{J}_p^{\text{locked}}$ وبصمته الرنينية الموزونة بالأهمية الدلالية $\mathbf{r}_p \in \mathbb{R}^V$، بحيث يتم توجيه الاستدلال بعد التجميد تلقائياً إلى الفضاء الجزئي الرنيني $p^* = \arg\max_p \sum_{t} r_p[x_t]$ دون أي نسيان كارثي عبر المراحل الأربع.

#### رابعاً: بوابة الجموح الانتقالية ثنائية المستوى (`Two-Tier Transition Refractory Gate`)
تُطبَّق أثناء التوليد الحر ([`generate(enable_refractory_gate=True)`](../models/model_scaled.py#L1439-L1507)):
1. **المستوى الأول — الجموح الفوري للرمز المتجاور ($t-1 \to t$):** يمنع أي رمز من تكرار نفسه في خطوتين متتاليتين مباشرة ($x_t \neq x_{t-1}$ عبر $\text{logits}[x_{t-1}] = -\infty$)، مما يقضي نهائياً على التلعثم الفوري (`"the the"` = **`0.00%`**) مع السماح الكامل بعودة الرمز طبيعياً عند $t+2, t+3$.
2. **المستوى الثاني — التكيف الانتقالي السياقي (`Contextual N-Gram Transition Adaptation`):** يثبط تكرار العبارات الرباعية (`4-grams`) والعبارات الثلاثية المحتوية على كلمات دلالية ($s(v) \ge 0.95$) عند محاولة تكرارها للمرة الثالثة، مما يقضي على الحلقات التكرارية في ذيل التوليد (`Phrase Loop Rate = 0.00%`) دون المساس بعودة الفاعل في الجملة الثانية بعد `". then"` و`". afterward"`.

---

### 5.3 النتائج المعملية المقارنة للطور 2.2 (`benchmarks/bench_v2_sleep_continual.py`)

#### الجدول (1): سجل التوليد العصبي، طور النوم المزدوج، وإعادة تدوير الخلايا (`TABLE 1: Phase-by-Phase Neurogenesis, Sleep Consolidation & Synaptic Recycling Log`)
| المرحلة (`Continual Phase`) | خلايا أُعيد إحياؤها موضعياً (`Rejuv In-Place`) | خلايا مدمجة (`Stage 1 Fused`) | خلايا مقلمة (`Stage 2 Pruned`) | خلايا مقفلة بالدرع (`Armor Locked`) | الأبعاد الفعالة (`v2.2 Effective Dims`) | أبعاد التدريب الساذج (`Naive Dims`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (Valley & Village)** | `0 cells` | `12 cells` | `36 cells` | `128 cells` | **`[32, 32, 32, 32]`** | `[160, 160, 160, 160]` |
| **Phase 2 (River & Lake)** | **`48 cells`** | `12 cells` | `36 cells` | `240 cells` | **`[60, 60, 60, 60]`** | `[256, 256, 256, 256]` |
| **Phase 3 (Forest & Meadow)** | **`48 cells`** | `12 cells` | `36 cells` | `352 cells` | **`[88, 88, 88, 88]`** | `[256, 256, 256, 256]` |
| **Phase 4 (Cottage & Mountain)** | **`48 cells`** | `12 cells` | `36 cells` | `464 cells` | **`[116, 116, 116, 116]`** | `[256, 256, 256, 256]` |
| **الإجمالي بعد التجميد (`Frozen W0`)** | **`144 Rejuvenated`** | **`48 Fused`** | **`144 Pruned`** | **`464 / 1,024 Active`** | **`54.7% Capacity Saved` (`4.86 MB`)** | `0.0% Saved` (`6.63 MB`) |

#### الجدول (2): مصفوفة الاحتفاظ بالذاكرة عبر 4 مراحل دون إعادة عرض (`TABLE 2: 4-Phase Zero-Replay Retention & Recall Matrix After Phase 4 — Frozen W0 Only`)
| المرحلة المقيمة بعد انتهاء المرحلة 4 | `Setting A: Naive Continual (No Sleep)` `PPL / Acc` | `Setting B: SS-ENN v2.2 (Sleep + Armor Lock)` `PPL / Acc` | مكسب الاحتفاظ (`Retention Gain`) |
| :--- | :---: | :---: | :---: |
| **Phase 1 (Valley & Village)** | `PPL: 5.86` \| `Acc: 66.44%` | **`PPL: 1.14`** \| **`Acc: 96.99%`** | **`+30.56%`** |
| **Phase 2 (River & Lake)** | `PPL: 3.18` \| `Acc: 84.03%` | **`PPL: 1.15`** \| **`Acc: 96.76%`** | **`+12.73%`** |
| **Phase 3 (Forest & Meadow)** | `PPL: 2.18` \| `Acc: 85.42%` | **`PPL: 1.12`** \| **`Acc: 97.92%`** | **`+12.50%`** |
| **Phase 4 (Cottage & Mountain)** | `PPL: 1.08` \| `Acc: 97.92%` | **`PPL: 1.16`** \| **`Acc: 97.22%`** | `-0.69%` |
| **المتوسط الشامل للمراحل الأربع (`4-Phase Mean`)** | `PPL: 3.08` \| `Acc: 83.45%` | **`PPL: 1.14`** \| **`Acc: 97.22%`** | **`+13.77%` (`-63.0% PPL`)** |

#### الجدول (3): تدقيق التوليد الحر (30 رمزاً)، ثبات الكيانات، وبوابة الجموح (`TABLE 3: 30-Token Rollout, Entity Consistency & Refractory Gate Audit — Frozen W0 Only`)
| `#` | `Seed Prompt` | `Entity Flow (Sent1 -> Bridge -> Sent2)` | `Adjacent Dups` | `Phrase Loops` | `Generated Rollout (+30 Tokens on Frozen W0)` |
| :-: | :--- | :---: | :---: | :---: | :--- |
| `01` | `"Once upon a time"` | `brave puppy -> (. then) -> brave puppy` (**`PASS`**) | **`0`** | **`0`** | `Once upon a time in the green valley, the brave puppy ran happily around the yard. then the brave puppy chased a colorful butterfly near the flowers and guarded the wooden gate` |
| `02` | `"The little bird"` | `little bird -> (. afterward) -> little bird` (**`PASS`**) | **`0`** | **`0`** | `The little bird sang a sweet song one sunny afternoon in the small village. afterward the little bird flew across the clear sky and built a safe nest in the branches.` |
| `03` | `"In the bright morning"` | `young rabbit -> (. then) -> young rabbit` (**`PASS`**) | **`0`** | **`0`** | `In the bright morning near the quiet river, the young rabbit hopped along the soft path. then the young rabbit found a fresh carrot in the grass and shared the food with` |
| `04` | `"The young rabbit"` | `young rabbit -> (. afterward) -> young rabbit` (**`PASS`**) | **`0`** | **`0`** | `The young rabbit hopped along the soft path in the bright morning near the quiet river. afterward the young rabbit found a fresh carrot in the grass and shared the food with` |
| `05` | `"Under the tall oak"` | `busy squirrel -> (. then) -> busy squirrel` (**`PASS`**) | **`0`** | **`0`** | `Under the tall oak tree in the ancient forest, the busy squirrel gathered shiny acorns for the winter. then the busy squirrel climbed up the tallest pine tree and stored the nuts` |
| **الإجمالي** | **Summary Audit** | **`100.00% (5/5) Entity Consistency`** | **`0.00% ('the the')`** | **`0.00% Loops`** | **`100% Clean Autoregressive Rollout on Compacted W0`** |

---

## 6. الطور 2.3: الصلابة الهرمية، تطبيع الطاقة الموضعي، والتعميم التركيبي الصفري (`Milestone 2.3: Hierarchical Stiffness Spectrum, Local Active RMSNorm & Role-Factored Compositional Generalization`)

بعد نجاح الطور 2.2 في التعلم المستمر متعدد المراحل، استهدف **الطور 2.3** كسر قيود القوالب المقترنة (`Template Coupling`) وتمكين النموذج المجمّد ($W_0$ فقط) من التعميم الصفري (`Zero-Shot Cross-Recombination`) على توليفات هجينة غير مرئية $(L_j + S_i + A_k)$ حيث $j \neq i \neq k$، دون أي اشتقاق عكسي أو قواعد نصية صلبة:

### 6.1 القوانين الفيزيائية الثلاثة للطور 2.3 (في [`models/model_scaled.py`](../models/model_scaled.py))
1. **طيف التنافسية والصلابة الهرمية (`Hierarchical Top-K & Stiffness Spectrum`):**
   - تتدرج عرضية التفعيل `Top-K` عبر العمق $l \in \{0, \dots, L-1\}$: تستخدم الطبقات الدنيا ($l < L/2$) نافذة أوسع $K_l = \min(D_{\text{active}}, \lfloor 1.5 K_{\text{base}} \rfloor)$ (`[24, 24]` في إعداد `1.5M`) لاحتضان التراكيب المألوفة وغير المألوفة معاً، بينما تتدرج الطبقات العليا نزولاً نحو `K_base` (`[20, 16]`) لانتقاء المفردات الدقيقة.
   - يتصاعد مضاعف الصلابة وسقفها غير المدرّع تدريجياً عبر العمق:
     $$\rho_l = \frac{l}{L - 1}, \quad \mu_k^{(l)} = 0.45 + 1.15 \rho_l, \quad k_{\max}^{(l)} = 0.28 + 0.57 \rho_l$$
     مما يُبقي الطبقات الدنيا مرنة لإعادة التشكيل التركيبي (`0.28 -> 0.47`) وتثبت الطبقات العليا المخرجات الدقيقة (`0.66 -> 0.85`).
2. **تطبيع الطاقة الجذري الموضعي الخالي من المعاملات (`Parameter-Free Local Active RMSNorm`):**
   - تُطبّع الطاقة الفعالة داخل الفضاء النشط لكل طبقة ديناميكية مباشرة دون أي معاملات قابلة للاشتقاق:
     $$\text{RMS}_{\text{sub}}(\mathbf{x}) = \sqrt{\frac{1}{D_{\text{act}}} \sum_{j=1}^{D_{\text{act}}} x_j^2 + \epsilon}, \qquad \hat{\mathbf{x}}_{:D_{\text{act}}} = \frac{1.20}{\text{RMS}_{\text{sub}}(\mathbf{x})} \mathbf{x}_{:D_{\text{act}}}$$
     مما يثبّت طاقة الإشارة عند **`1.2000`** عبر جميع الطبقات على السلاسل التدريبية والهجينة على حد سواء.
3. **فصل مسار الكيان عن المشهد ذاتياً عبر الأدوار (`Unsupervised Role-Factored Subject-Scene Decoupling`):**
   - في `calibrate_salience_from_tokens()`، تُكتشف مفردات الكيانات المتكررة عبر الجمل (`is_entity_vocab`) تلقائياً من تكرر الثنائيات الانتقائية (`DF < 30%`) عبر فجوة جُملية ($\Delta t \ge 4$).
   - يقرأ رأس الأدوار [`role_bridge_head`](../models/model_scaled.py) النافذة السببية الرباعية مع الانحياز الافتراضي للمشهد (`[lag3, lag2, lag1, tok, 1.0]`) ويتعلم محلياً فصل أدوار `SCENE` عن استدعاء الكيان `ENT_START / ENT_END` عبر توازن الفئات النادرة، بينما يوجه [`entity_proj`](../models/model_scaled.py) المتجه السببي للكيان المكتمل إلى فضاء تضمين المفردات (`W0` فقط بعد التجميد).

### 6.2 نتائج الاختبار المعياري للطور 2.3 (`benchmarks/bench_v2_compositional_gen.py` — `Frozen W0 Only`)
*(Split: 64 Diagonal Train $(L_i, S_i, A_i)$ vs. 24 Unseen Cross-Recombinations $(L_j, S_i, A_k)$ with $j \neq i \neq k$ | AST: `25` files, `0` autograd)*

#### الجدول (1): دقة التعميم الصفري والربط المضاد للقالب (`Zero-Shot Cross-Recombination & Subject Binding`)
| المعمارية (`Architecture — Frozen W0`) | استرجاع التدريب (`Train Rec%`) | دقة الهجين الكاملة (`Hybrid Full%`) | دقة الاستمرار التركيبي (`Hybrid Cont%`) | حيرة الهجين (`Hybrid PPL`) | ربط الفاعل بالمضاد (`Subj Bind%`) | الانجراف للمالك الأصلي (`Owner Rev%`) | زمن التدريب (`Time`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Baseline Flat (v2.1)`** | `95.68%` | `89.21%` | `95.78%` | `1.801` | `95.83%` | `0.00%` | `5.52s` |
| **`SS-ENN v2.3 (Hier + ActiveRMS + Role)`** | **`95.62%`** | **`90.27%`** | **`97.01%`** | `1.870` | **`100.00%` (`48/48`)** | **`0.00%`** | `7.75s` |

#### الجدول (2): التدرج الطبقي لـ `Top-K` والصلابة $\bar{k}_l$ وتثبيت طاقة `RMSNorm` (`L1..L4`)
| الطبقة | `Flat Top-K` | `v2.3 Top-K` | `Flat k_bar(max)` | `v2.3 k_bar(max)` | `Flat Hybrid RMS` (منجرف) | `v2.3 Pre -> Post Active-RMS` (ثابت) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **L1** | `16` | **`24`** | `0.0949 (0.850)` | **`0.0799 (0.280)`** | `1.5375` | **`1.5291 -> 1.2000`** |
| **L2** | `16` | **`24`** | `0.0973 (0.850)` | **`0.1187 (0.470)`** | `1.6025` | **`1.2682 -> 1.2000`** |
| **L3** | `16` | **`20`** | `0.1044 (0.850)` | **`0.1550 (0.660)`** | `1.6743` | **`1.2629 -> 1.2000`** |
| **L4** | `16` | **`16`** | `0.1017 (0.850)` | **`0.1676 (0.850)`** | `1.7607` | **`1.2620 -> 1.2000`** |

#### الجدول (3): تدقيق التوليد الحر (+30 رمزاً على `W0`) لـ 6 بادئات هجينة متقاطعة
*النتيجة الإجمالية: **`100.00% (6/6 PASS)`** ثبات فاعل هجين في الجملة الثانية (`0.00%` انجراف لمالك القالب)، **`0.00%`** تكرار متجاور، و**`0.00%`** حلقات عبارة ثلاثية:*
- **`H1..H3` (جسور الجملة الثانية `afterward / then / soon` في بيئات مضادة):** التزمت جميعها بنسبة 100% بالفاعل المحقون (`little fox`, `brave puppy`, `tiny squirrel`) مع متابعة نحوية سليمة.
- **`H4..H6` (إكمال حر من الجملة الأولى `Meadow+Rabbit`, `Canyon+Duck`, `Hollow+Owl`):** أكملت الجملة الأولى بسلاسة عبر جسر `soon / happily` واحتفظت بنسبة 100% بالفاعل الهجين في الجملة الثانية.

---

## 7. الطور 2.4: الكبسولات المرنة المتفرقة كتلياً، الممر السريع الموزون بالعمق، والتوسع العميق لـ 12 طبقة (`Milestone 2.4: Block-Sparse Elastic Capsules, Depth-Scaled Residual Highway & 12-Layer Scaling`)

بعد نجاح الطور 2.3 على الشبكات ذات الطبقات الأربع، عالج **الطور 2.4** عقبتين هيكليتين عند الانتقال نحو العمق الكبير (`L = 12` طبقة) وكفاءة تنفيذ الذاكرة على المعالج:
1. استبدال التنافس المبعثر للخلايا (`Element-wise Top-K`) بالتنافس الكتلي المتجاور (`Contiguous Block-Sparse Capsules`) مع تنفيذ مقصوص للأوزان الفعالة (`Active-Slice Contiguous Execution`).
2. الحفاظ على تباين الإشارة (`Variance Ratio ~ 1.00`) وتوجيه الخطأ المباشر (`DFA`) دون تلاشي أو تضخم عبر 12 طبقة دون أي انتشار عكسي (`Zero-Autograd`).

### 7.1 القوانين الفيزيائية الأربعة للطور 2.4 (في [`models/model_scaled.py`](../models/model_scaled.py))
1. **القانون الأول — التنافس الكتلي المتجاور وموازنة الحمل (`Contiguous Block-Sparse Capsules & Load-Balanced Routing`):**
   - يُقسّم الفضاء النشط $D_{\text{active}}^{(l)}$ (المتوافق صارماً مع مضاعفات `block_size = 16`) إلى $N_l = D_{\text{active}}^{(l)} / 16$ كبسولة متجاورة في الذاكرة عبر `_forward_block_sparse_layer()` و `_update_block_sparse_layer()`.
   - تُرتب الكبسولات بحسب إجمالي طاقة الاستجابة $\sum_{j \in \mathcal{B}_b} |h_j|^2$ معدّلاً بمؤشر الاستخدام طويل الأمد لمنع احتكار الكبسولات (`block_balance_strength = 0.68`)، مع انتقاء أعلى $K_{\text{blocks}}^{(l)}$ (`hierarchical_top_k_blocks`: `2` بلوكات في الطبقات المبكرة، ثم `1` بلوك في الطبقات الأعمق).
2. **القانون الثاني — الممر المتبقي الموزون بالعمق (`Depth-Scaled Residual Highway`):**
   - بعد تطبيع الممر الأولي عبر `_apply_subspace_rmsnorm(a, target_rms=1.20)`، تمر الإشارة عبر كل طبقة وفق:
     $$\mathbf{h}^{(l)} = \text{RMSNorm}_{\text{active}}\!\left(\mathbf{h}^{(l-1)} + \frac{\alpha_{\text{res}}}{\sqrt{L}} \, \mathbf{a}_{\text{block}}^{(l)}\right), \qquad \alpha_{\text{res}} = 1.35$$
     ما يُبقي نسبة تباين الإشارة `RMS(L12) / RMS(L1)` عند **`1.0000`** (`Post-RMS`) و **`0.9901`** (`Pre-RMS`) بدلاً من التضخم غير المضبوط `1.2225`.
3. **القانون الثالث — التوجيه العميق المرسّى بالممر السريع (`Highway-Anchored Deep Hebbian DFA`):**
   - تتكامل مصفوفة التوجيه المباشر $B^{(l)}$ مع الإسقاط الصفّي المطبّع لرأس الإخراج $W_{\text{head}}^T$ (`dfa_anchor_blend = 0.65`) إلى جانب أثر هيبي موضعي عبر `_sync_highway_anchored_dfa()`، مع استنساخ مرجعية `anchor_unit_full` مرة واحدة لكل دفعة لتجنب حسابها 12 مرة لكل طبقة، وتدمير $B^{(l)}$ كلياً عند التجميد (`production_freeze()`).
4. **القانون الرابع — التنفيذ الكتلي المقصوص للذاكرة الفعالة (`Active-Slice Contiguous Execution & Causal Top-K Vectorization`):**
   - استبدال حلقة التفرق السببي في [`DualHorizonCausalAttention`](../models/model_scaled.py) بانتقاء متجهي (`vectorized_topk = True`, `Future Leakage = 0.0`).
   - تنفيذ ضرب المصفوفات وتحديث النوابض على الشريحة النشطة المتصلة فقط `W_0[:d_active, :d_eff]`، وقصّ مساقط السياق الخطية (`cap_proj`, `lag1_proj`, `lag2_proj`) عند التجميد إلى `[:d_eff, :]` مع الاحتفاظ بأوزان $W_0$ المصفّاة وفق محاذاة `block_size = 16`.

### 7.2 نتائج الاختبار المعياري للطور 2.4 (`benchmarks/bench_v2_deep_block_sparse.py` — `Frozen W0 Only`)
*(AST: `26` Python files scanned, `0 backward/grad` | English Split: `Train=355 (12,780 tok)` / `Val=39 (1,404 tok)` | `V=2,048`)*

#### الجدول (1): المقارنة الرباعية للأداء والسرعة والاستقرار على `CPU` (`Frozen W0 Only`)
| المعمارية (`Architecture`) | دقة التدريب (`Train%`) | دقة التحقق (`Val%`) | الحيرة (`ValPPL`) | سرعة التدريب (`tok/s`) | سرعة التجميد (`Eval tok/s`) | حجم $W_0$ (`MB`) | متوسط الصلابة ($\bar{k}$) | نسبة التباين (`L_N/L1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`4L Element-wise Sparse`** | `93.11%` | `93.09%` | `1.54` | `8,875` | `31,273` | `5.61` | `0.1839` | `1.0000` |
| **`4L Block-Sparse (B=16)`** | **`93.76%`** | **`93.73%`** | **`1.45`** | **`10,920` (`1.23x`)** | **`41,862` (`1.34x`)** | **`5.42`** | **`0.2027`** | **`1.0000`** |
| **`12L Deep Naive (Unscaled/Rand)`** | `93.44%` | `92.88%` | `1.44` | `7,665` | `24,109` | `6.54` | `0.0238` (منهار) | `1.2225` (منجرف) |
| **`12L Deep SS-ENN v2.4 (B=16)`** | **`93.66%`** | **`93.59%`** | **`1.48`** | `5,443` | `20,516` | `6.54` | **`0.1733`** | **`1.0000`** |

#### الجدول (2): التوصيف الطبقي التفصيلي لشبكة الـ 12 طبقة (`12L Deep SS-ENN v2.4 — L1..L12`)
*إجمالي الإنتروبيا الكتلية لـ 12 طبقة: التدريب **`0.9864`** | التقييم المجمد **`0.9133`** || نسبة التباين `L12/L1`: `Post = 1.0000` / `Pre = 0.9901`:*

| الطبقة | الأبعاد النشطة | الكبسولات (`K/N x16`) | `Pre-RMS` | `Post-RMS` | `L_l/L1` | متوسط الصلابة ($\bar{k}_l$) | إنتروبيا الكبسولات (`Caps Ent`) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **`L1`** | `192` | `2 / 12 (x16)` | `1.2244` | `1.2000` | `1.0000` | `0.1037` | `0.9134` |
| **`L2`** | `192` | `2 / 12 (x16)` | `1.2289` | `1.2000` | `1.0000` | `0.1265` | `0.9037` |
| **`L3`** | `192` | `2 / 12 (x16)` | `1.2304` | `1.2000` | `1.0000` | `0.1435` | `0.9056` |
| **`L4`** | `192` | `2 / 12 (x16)` | `1.2344` | `1.2000` | `1.0000` | `0.1623` | `0.8960` |
| **`L5`** | `192` | `2 / 12 (x16)` | `1.2395` | `1.2000` | `1.0000` | `0.1793` | `0.9280` |
| **`L6`** | `192` | `2 / 12 (x16)` | `1.2487` | `1.2000` | `1.0000` | `0.1995` | `0.8575` |
| **`L7`** | `192` | `1 / 12 (x16)` | `1.2195` | `1.2000` | `1.0000` | `0.1672` | `0.9304` |
| **`L8`** | `192` | `1 / 12 (x16)` | `1.2149` | `1.2000` | `1.0000` | `0.1767` | `0.9056` |
| **`L9`** | `192` | `1 / 12 (x16)` | `1.2124` | `1.2000` | `1.0000` | `0.1853` | `0.9578` |
| **`L10`** | `192` | `1 / 12 (x16)` | `1.2192` | `1.2000` | `1.0000` | `0.2025` | `0.9373` |
| **`L11`** | `192` | `1 / 12 (x16)` | `1.2204` | `1.2000` | `1.0000` | `0.2151` | `0.8920` |
| **`L12`** | `192` | `1 / 12 (x16)` | `1.2123` | `1.2000` | `1.0000` | `0.2185` | `0.9330` |

#### الجدول (3): تدقيق التوليد الحر (+30 رمزاً على `W0` المجمد لـ 12 طبقة)
- **ثبات الكيان عبر الجمل (`Entity Consistency`):** **`100.00% (5/5 PASS)`** (`Once upon a time`, `The little bird`, `In the bright morning`, `The young rabbit`, `Under the tall oak`).
- **نسبة التكرار المتجاور (`Adjacent Dup`):** **`0.00%`**.
- **نسبة حلقات العبارة (`Phrase Loops`):** **`0.00%`**.

---

## 8. طريقة إعادة الإنتاج والتحقق (`Reproducibility`)
```bash
# 1. تشغيل معيار الطور 2.4 للكبسولات الكتلية المتفرقة (B=16) والتوسع العميق لـ 12 طبقة
python benchmarks/bench_v2_deep_block_sparse.py

# 2. تشغيل معيار الطور 2.3 للتعميم التركيبي الصفري، الصلابة الهرمية، وتطبيع الطاقة الموضعي
python benchmarks/bench_v2_compositional_gen.py

# 3. تشغيل معيار الطور 2.2 للتعلم المستمر عبر 4 مراحل، طور النوم المزدوج، وإعادة تدوير الخلايا
python benchmarks/bench_v2_sleep_continual.py

# 4. تشغيل معيار الطور 2.1 للمحرك المليوني والترقية الفيزيائية على المعالج المحلي
python benchmarks/bench_scale_english_cpu.py

# 5. تشغيل حزمة الاختبارات الصارمة (8 اختبارات) وفحص AST عبر جميع ملفات المستودع (26 ملفاً برمجياً)
python tests/test_suite.py
```
