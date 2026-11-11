# Self-Sizing Elastic Neural Networks (SS-ENN)
**شبكات النوابض المرنة ذاتية التشكيل — إطار تعلم موضعي خالٍ من الانتشار العكسي (`Zero-Autograd`) للتوليد المتسلسل والتعلم المستمر**

- **الورقة البحثية الكاملة (Full Technical Research Paper):** [PAPER_SS_ENN.md](PAPER_SS_ENN.md)
- **المواصفات الهندسية الأساسية (Architecture Spec):** [SPEC_SPRING_MVP.md](SPEC_SPRING_MVP.md)

## بنية المستودع (Repository Structure)
- `engine/spring_layer.py`: طبقة النوابض الأساسية $(w, w_0, k)$ مع قناع `Top-K` والقوانين الفيزيائية الثلاثة.
- `engine/dynamic_spring_layer.py`: طبقة التوليد العصبي الديناميكي وحوض الخلايا النائمة (`Dormant Reserve & Dynamic Neurogenesis`).
- `engine/sparse_attention.py`: طبقة الانتباه الذاتي السببي المتفرق (`Sparse Causal Self-Attention`) الخالية من الاشتقاق العكسي.
- `engine/dfa_router.py`: مصفوفات التوجيه المباشر للخطأ (`Direct Feedback Alignment`).
- `engine/causal_context.py`: معالج السياق السببي المقنّع.
- `model.py` / `model_v2.py` / `model_dynamic.py` / `model_continuous.py`: نماذج المراحل الأربع.
- `benchmarks/`:
  - `bench_capacity.py`: اختبار سقف السعة والتشبع ($12 \to 25 \to 50$ جملة).
  - `bench_continual.py`: اختبار التعلم المستمر ومقاومة النسيان الكارثي (`Zero-Replay`).
  - `bench_sparse_attention.py`: اختبار الانتباه السببي المتفرق على الجمل الطويلة (6 كلمات).
  - `bench_neurogenesis.py`: اختبار التوليد العصبي الديناميكي وتقليم الخلايا النائمة على 100 جملة (`tier_4_ultra`).
  - `bench_continuous_stream.py`: اختبار الفقرات النصية المتصلة والتوليد الذاتي الحر (`Free-Running Rollout`).
- `test_suite.py`: حزمة التحقق الصارم وفحص الشجرة النحوية المجردة (`AST Zero-Autograd Verification`).
