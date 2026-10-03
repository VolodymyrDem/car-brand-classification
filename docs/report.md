# Звіт: визначення марки автомобіля за фотографією

**Дисципліна:** ШІ: принципи та методи
**Виконав:** Володимир Демянчук
**Репозиторій:** https://github.com/VolodymyrDem/car-brand-classification

> ⚠️ Чернетка. Місця з позначкою **TODO** заповнюються після навчання моделей у Google Colab.

---

## 1. Постановка задачі

Задача — **багатокласова класифікація зображень**: за фотографією легкового автомобіля визначити його
марку (BMW, Audi, Chevrolet, ...). Вхід — кольорове RGB-зображення довільного розміру, вихід — одна з 20 марок.

Мета роботи — розв'язати задачу моделями трьох рівнів складності та порівняти їх:

| Рівень | Підхід | Модель |
|---|---|---|
| 1 | Класичне машинне навчання на ручних ознаках | Logistic Regression, Random Forest (scikit-learn) |
| 2 | Згорткова нейромережа (CNN), fine-tuning | ResNet18 (попередньо навчена на ImageNet) |
| 3 | Трансформер, fine-tuning | Vision Transformer ViT-Small/16 (ImageNet-21k → ImageNet-1k) |

Чому задача складна:

- різні моделі однієї марки можуть сильно відрізнятися (пікап і седан Chevrolet), а моделі різних
  марок — бути схожими (Chevrolet і GMC мають спільні платформи);
- фото зроблено з різних ракурсів, на різному фоні й освітленні, часто з водяними знаками;
- автомобіль займає різну частку кадру.

## 2. Дані

### 2.1. Вибір датасету

Використано **Stanford Cars** (Krause et al., 2013) — 16 185 фото, 196 класів виду
«марка + модель + рік». Датасет завантажено з дзеркала на Hugging Face (`tanganke/stanford_cars`),
оскільки оригінальні посилання Стенфорда більше не працюють. Марку виділено з назви класу
(з урахуванням багатослівних марок: *AM General*, *Aston Martin*, *Land Rover*).

### 2.2. Фільтрація класів

Після групування за марками отримано 49 марок, але більшість із них має менше 100 фото, що недостатньо
для навчання та чесного тестування. Тому залишено лише марки з **≥ 300 зображень**:

| Показник | Значення |
|---|---|
| Марок (класів) | 20 |
| Зображень | 12 681 |
| Моделей авто всередині цих марок | 153 |
| Співвідношення найбільшого / найменшого класу | 5.73 (Chevrolet / Buick) |
| Медіанний розмір фото | 640 × 426 |

![Розподіл фото за марками](../reports/figures/01_all_makes_distribution.png)

Оригінальне розбиття Stanford Cars на train/test **не використано**: обидві частини об'єднано і заново
розбито так, щоб мати окрему валідаційну вибірку. Рамки (bounding boxes) у дзеркалі відсутні, тому
фото **не обрізаються** по автомобілю — моделі бачать увесь кадр, включно з фоном. Це ускладнює задачу,
але ближче до реального застосування.

### 2.3. Розбиття train / val / test

Стратифіковане розбиття 70 / 15 / 15 % з фіксованим `seed = 42` (частка кожної марки однакова в усіх трьох частинах):

| Вибірка | Фото | Призначення |
|---|---|---|
| train | 8 876 | навчання |
| val | 1 902 | вибір гіперпараметрів / рання зупинка для нейромереж |
| test | 1 903 | **одноразова** фінальна оцінка всіх моделей |

![Розподіл за вибірками](../reports/figures/02_split_distribution.png)

Для рівня 1 використано **стратифіковану 5-fold крос-валідацію** на train + val (10 778 фото):
кожна комбінація гіперпараметрів навчається 5 разів на 4/5 даних і оцінюється на решті 1/5.
Для нейромереж крос-валідація занадто дорога (5× час навчання на GPU), тому використано фіксовану val-вибірку.
Test-вибірка однакова для всіх моделей і не використовувалася ні для навчання, ні для вибору гіперпараметрів.

### 2.4. Аналіз даних

![Приклади фото](../reports/figures/03_sample_images.png)

![Розміри фото](../reports/figures/04_image_sizes.png)

Детальний аналіз — у ноутбуці [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).

## 3. Методи

### 3.1. Метрики

- **Accuracy** — частка правильних відповідей.
- **Macro F1** — середнє F1 по всіх 20 марках з однаковою вагою; **основна метрика**, бо класи незбалансовані.
- **Top-3 accuracy** — правильна марка серед трьох найімовірніших.
- Час навчання, час передбачення на одне фото, розмір моделі.

### 3.2. Рівень 1 — класичне ML (scikit-learn)

Класичні моделі не вміють працювати з пікселями напряму, тому з кожного фото витягується вектор
**ручних ознак** довжиною 2 820:

- **HOG** (Histogram of Oriented Gradients, Dalal & Triggs, 2005) — 2 772 числа: фото зменшується до
  192 × 128 у відтінках сірого, ділиться на клітинки 16 × 16, у кожній рахується гістограма 9 напрямків
  градієнтів, блоки 2 × 2 нормалізуються (L2-Hys). Описує **форму** (контури кузова, фар, решітки).
- **Гістограма кольору HSV** — 3 × 16 = 48 чисел. Описує **колір**.

![HOG-ознаки](../reports/figures/05_hog_features.png)

**Logistic (softmax) Regression.** Конвеєр `StandardScaler → PCA → LogisticRegression`.
PCA зменшує розмірність і прибирає шум. Сітка гіперпараметрів (12 комбінацій × 5 фолдів = 60 навчань):

| Параметр | Значення |
|---|---|
| `pca__n_components` | 128, 512 |
| `C` (обернена сила L2-регуляризації) | 0.001, 0.01, 0.1 |
| `class_weight` | None, balanced |

**Random Forest** (Breiman, 2001) — ансамбль з 300 дерев рішень. Сітка (4 комбінації × 5 фолдів):

| Параметр | Значення |
|---|---|
| `n_estimators` | 300 |
| `max_features` | sqrt, log2 |
| `class_weight` | None, balanced_subsample |

Критерій вибору — macro F1 на крос-валідації. Найкраща комбінація перенавчається на всіх 10 778 фото
і один раз оцінюється на test.

### 3.3. Рівень 2 — ResNet18

ResNet (He et al., 2015) — згорткова мережа з залишковими (skip) з'єднаннями, які дозволяють
навчати глибокі мережі. ResNet18 має 18 шарів і ~11.2 млн параметрів. Використано ваги, попередньо
навчені на ImageNet (`resnet18.a1_in1k` з бібліотеки timm), останній шар замінено на новий з 20 виходами
(**transfer learning / fine-tuning**). Мережа сама вчиться виділяти ознаки — не потрібен ручний HOG.

### 3.4. Рівень 3 — Vision Transformer

ViT (Dosovitskiy et al., 2020) ділить фото 224 × 224 на 196 патчів 16 × 16, перетворює кожен на
вектор (токен) і обробляє їх 12 блоками self-attention: кожен патч «дивиться» на всі інші, тому модель
одразу бачить глобальний контекст (на відміну від локальних згорток CNN).
Використано ViT-Small/16 (~21.7 млн параметрів), навчений на ImageNet-21k і дотренований на ImageNet-1k
(`vit_small_patch16_224.augreg_in21k_ft_in1k`).

### 3.5. Налаштування навчання нейромереж

| Параметр | ResNet18 | ViT-Small |
|---|---|---|
| Розмір входу | 224 × 224 | 224 × 224 |
| Оптимізатор | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 |
| Learning rate (backbone / голова) | 1e-3 / 3e-3 | 1e-4 / 1e-3 |
| Розклад LR | warmup 1 епоха + cosine | warmup 1 епоха + cosine |
| Макс. епох | 20 | 15 |
| Batch size | 64 | 64 |
| Втрати | Cross-entropy, label smoothing 0.1 | те саме |
| Рання зупинка | 5 епох без покращення val macro F1 | те саме |

- **Заморожування:** перші 2 епохи навчається лише нова голова (backbone заморожений), щоб випадкові
  ваги голови не «зламали» попередньо навчені ознаки; далі — fine-tuning усієї мережі.
- **Аугментація** (лише train): випадкова обрізка (60–100 % площі, пропорції 1.2–1.8), горизонтальне
  віддзеркалення, зміна яскравості / контрасту / насиченості.
- Mixed precision (fp16) на GPU, зберігається модель з найкращим val macro F1.

![Аугментація](../reports/figures/06_augmentations.png)

### 3.6. Середовище

- Рівні 1–3 навчено в **Google Colab** (GPU NVIDIA T4, 2 vCPU); попередня обробка даних і EDA — локально (Apple M4 Pro).
- Python 3.12, PyTorch, timm, scikit-learn, scikit-image, pytorch-grad-cam.

## 4. Результати

### 4.1. Порівняльна таблиця (test, 1 903 фото)

**TODO:** вставити `reports/tables/comparison.md`.

| Модель | Рівень | Accuracy | Top-3 | Macro F1 | Час навчання | мс / фото | Розмір, МБ |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| Random Forest | 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| ResNet18 | 2 | TODO | TODO | TODO | TODO | TODO | TODO |
| ViT-Small/16 | 3 | TODO | TODO | TODO | TODO | TODO | TODO |

![Порівняння метрик](../reports/figures/comparison_metrics.png)

![Якість vs час](../reports/figures/comparison_quality_vs_time.png)

### 4.2. Рівень 1: крос-валідація

**TODO:** найкращі параметри та CV macro F1 (mean ± std) для LR і RF з `metrics_*.json`.

Попередній локальний запуск Logistic Regression: найкращі параметри `C = 0.01`, PCA 512, без зважування
класів; CV macro F1 = 0.168 ± 0.005; на test — accuracy 20.1 %, top-3 42.7 %, macro F1 0.162
(випадкове вгадування — 5 %).

### 4.3. Криві навчання нейромереж

![ResNet18](../reports/figures/learning_curves_resnet18.png)

![ViT](../reports/figures/learning_curves_vit_small.png)

**TODO:** коментар — на якій епосі найкращий val F1, чи є перенавчання (розрив train/val loss).

### 4.4. Якість по марках

![F1 по марках](../reports/figures/comparison_per_class_f1.png)

**TODO:** які марки розпізнаються найкраще / найгірше.

### 4.5. Матриці помилок

**TODO:** вставити `confusion_matrix_*.png` та прокоментувати найчастіші пари помилок
(`top_confusions_*.csv`; очікувано Chevrolet ↔ GMC, Dodge ↔ Chrysler).

### 4.6. Інтерпретація: куди дивляться моделі

![Grad-CAM та attention rollout](../reports/figures/gradcam_examples.png)

- **Grad-CAM** (Selvaraju et al., 2017) показує, які ділянки фото найбільше вплинули на передбачення.
- **Attention rollout** (Abnar & Zuidema, 2020) показує, на які патчі дивиться токен класу ViT.

**TODO:** чи дивляться моделі на решітку, фари, логотип, чи на фон.

![Найвпевненіші помилки ResNet18](../reports/figures/misclassified_resnet18.png)

![Найвпевненіші помилки ViT](../reports/figures/misclassified_vit_small.png)

## 5. Обговорення

**TODO** після отримання результатів. План:

- Чому класичне ML слабке: HOG описує загальну форму, яка більше залежить від **типу кузова і ракурсу**,
  ніж від марки; дрібні деталі (логотип, решітка) губляться при 192 × 128 і клітинках 16 × 16.
- Наскільки transfer learning покращує результат і чому (ознаки, вивчені на мільйонах фото ImageNet).
- ResNet vs ViT: різниця в якості, швидкості та розмірі; роль попереднього навчання на ImageNet-21k.
- Вплив дисбалансу класів, найважчі марки.
- Обмеження: немає обрізки по авто, лише 20 марок, роки випуску до 2012, одне розбиття для нейромереж.
- Можливі покращення: обрізка детектором (YOLO), більші моделі / роздільність, TTA, ансамбль.

## 6. Висновки

**TODO.**

## 7. Як відтворити

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # завантаження і розбиття даних
python scripts/eda.py                   # графіки EDA
python scripts/train_classical.py       # рівень 1
python scripts/train_deep.py --model resnet18    # рівень 2 (GPU)
python scripts/train_deep.py --model vit_small   # рівень 3 (GPU)
python scripts/compare.py               # порівняльна таблиця і графіки
python scripts/interpret.py             # Grad-CAM, attention rollout, аналіз помилок
```

Або все одразу в Colab: [`notebooks/colab_train.ipynb`](../notebooks/colab_train.ipynb).

## 8. Джерела

1. Krause J., Stark M., Deng J., Fei-Fei L. *3D Object Representations for Fine-Grained Categorization.* ICCV Workshops, 2013.
2. Dalal N., Triggs B. *Histograms of Oriented Gradients for Human Detection.* CVPR, 2005.
3. Breiman L. *Random Forests.* Machine Learning, 45(1), 2001.
4. He K., Zhang X., Ren S., Sun J. *Deep Residual Learning for Image Recognition.* arXiv:1512.03385, 2015.
5. Dosovitskiy A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* arXiv:2010.11929, 2020.
6. Steiner A. et al. *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.* arXiv:2106.10270, 2021.
7. Selvaraju R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
8. Abnar S., Zuidema W. *Quantifying Attention Flow in Transformers.* ACL, 2020.
9. Wightman R. *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models
