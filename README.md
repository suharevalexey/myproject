# Лабораторные работы по курсу MLOps

Репозиторий содержит практические задания по настройке рабочего окружения, линтеров, версионированию кода, данных и ML-пайплайнов.

---

## Структура проекта

```text
├── .dvc/                  # Конфигурация и метаданные DVC
├── data/                  # Данные (хранятся в DVC, игнорируются Git)
│   └── raw_data.csv
├── output/                # Результаты работы пайплайна
│   └── metrics.txt        # Метрики пайплайна
├── src/
│   ├── myproject/         # Исходный код проекта (ДЗ №1)
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── train.py
│   ├── generate_data.py   # Скрипт генерации синтетического датасета (ДЗ №3)
│   └── process_data.py    # Скрипт обработки и подсчёта метрик (ДЗ №3)
├── tests/                 # Пирамида тестов (Unit, Property, Smoke, Overfit, E2E)
│   ├── conftest.py
│   ├── test_block.py
│   ├── test_dataloader.py
│   ├── test_e2e.py
│   ├── test_model.py
│   └── test_train.py
├── .dvcignore
├── .gitignore
├── .python-version
├── dvc.yaml               # Описание стадий пайплайна DVC
├── dvc.lock               # Зафиксированные контрольные суммы артефактов DVC
├── pyproject.toml         # Зависимости uv и настройки Ruff / ty
├── uv.lock                # Точные версии зависимостей Python
└── README.md
```

---

## Быстрый старт

Проект управляется через пакетный менеджер **[uv](https://docs.astral.sh/uv/)**.

```powershell
# Клонирование репозитория
git clone https://github.com/suharevalexey/MLOps.git
cd MLOps

# Установка окружения и зависимостей
uv sync
```

---

## Домашнее задание №1. Настройка линтеров и типизации (uv, Ruff, ty)

В рамках первого задания настроено изолированное окружение через `uv` и подключены инструменты статического анализа кода: **Ruff** (линтер + форматтер) и **ty** (тайпчекер).

### Настройки инструментов (`pyproject.toml`):
* `tool.ruff`: поддержка Python 3.13, ограничение длины строки — 88 символов.
* `tool.ruff.lint.select`: правила `E` (стиль PEP 8), `F` (логические ошибки Pyflakes), `I` (isort — сортировка импортов).
* `tool.ty.environment`: проверка окружения Python 3.13.
* `tool.ty.rules`: строгость `error` для ключевых проверок типов.

### Запуск проверок:
```powershell
# Проверка качества кода линтером
uv run ruff check .

# Проверка форматирования
uv run ruff format --check .

# Статический анализ типов
uv run ty check
```

---

## Домашнее задание №2. Работа с Git

Отработаны базовые практики работы с распределённой системой контроля версий:
* создание feature-веток;
* решение конфликтов слияния (merge conflicts);
* оформление атомарных коммитов согласно conventional commits;
* синхронизация изменений с удаленным репозиторием GitHub.

---

## Домашнее задание №3. Версионирование данных и пайплайн (DVC)

В рамках задания настроена интеграция Data Version Control (DVC) для трекинга артефактов и автоматизации запуска этапов обработки данных.

### 1. Настройка удалённого хранилища (Remote)
Для изолированного хранения артефактов данных настроен локальный remote вне репозитория:
```powershell
# Регистрация локального remote по умолчанию
uv run dvc remote add -d mylocalremote ~/.dvc_local_storage
```

### 2. Пайплайн обработки (`dvc.yaml`)
Пайплайн состоит из двух независимых стадий:

1. **`generate_data`** (генерация данных):
   * **Команда:** `uv run python src/generate_data.py`
   * **Зависимости (`deps`):** `src/generate_data.py`
   * **Выход (`outs`):** `data/raw_data.csv` (файл данных версионируется DVC и автоматически внесён в `.gitignore`).

2. **`process_data`** (обработка и валидация):
   * **Команда:** `uv run python src/process_data.py`
   * **Зависимости (`deps`):** `src/process_data.py`, `data/raw_data.csv`
   * **Метрики (`metrics`):** `output/metrics.txt` (сохраняет результат подсчёта обработанных строк).

### 3. Воспроизведение и синхронизация

```powershell
# Запуск воспроизведения пайплайна
uv run dvc repro

# Просмотр зафиксированных метрик
uv run dvc metrics show

# Отправка данных в хранилище (DVC Remote)
uv run dvc push

# Скачивание данных из хранилища (при переносе на новую машину)
uv run dvc pull
```

---

## Домашнее задание №4. Тестирование ML-пайплайна (TDD, pytest, Hypothesis)

В рамках 3–4 лекций реализована пирамида тестирования машинного обучения по методологии **TDD (Test-Driven Development: Red → Green)**.

### 1. Архитектура пайплайна (`src/myproject/`)
* **`models.py`**: полносвязный блок `LinearBlock` (Linear + ReLU) и классификатор `SimpleClassifier`.
* **`dataset.py`**: воспроизводимый датасет `SyntheticDataset` и утилита сборки батчей `create_dataloader`.
* **`trainer.py`**: класс `Trainer`, инкапсулирующий шаги forward/backward/step, поддержку сухого прогона (`fast_dev_run`), smoke-проверку градиентов и переобучение одного батча (`overfit_batch`).

### 2. Структура тестов и метки (`tests/`)
Зарегистрированы и покрыты следующие категории тестов (`pytest marks`):
* **`@pytest.mark.unit`**: изолированное тестирование компонентов (размерности выходов, расчет метрик, сборка батчей).
* **`@pytest.mark.property`**: property-based тестирование инвариантов формы (shape preservation) при произвольных размерах батча с помощью библиотеки `Hypothesis`.
* **`@pytest.mark.smoke`**:
  * *Dry run*: 1–2 батча без падений цикла оптимизации (`fast_dev_run=True`).
  * *Smoke-тест*: проверка конечности loss, убывания ошибки и обязательного наличия `param.grad is not None` у всех обучаемых параметров.
* **`@pytest.mark.overfit`**: тест сходимости на одном батче до почти нулевого loss ($< 0.05$).
* **`@pytest.mark.e2e`**: сквозное тестирование полного цикла обучения и последующего инференса.

### 3. Запуск проверок
```powershell
# Запуск всех тестов с отчётом о покрытии кода
uv run pytest -v --cov=myproject --cov-report=term-missing

# Запуск тестов по конкретной метке (например, smoke или property)
uv run pytest -m smoke -v
uv run pytest -m property -v
```
