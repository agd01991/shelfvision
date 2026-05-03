# Установка зависимостей ShelfVision через WSL

Для установки всех зависимостей в Linux-виртуальную среду через WSL используется отдельная среда:

```text
.venv_wsl
```

Она отделена от Windows-среды `.venv`, чтобы не смешивать Windows- и Linux-пакеты.

## Через Control Panel

1. Запустить панель управления:

```bat
scripts\windows\start_control_panel.bat
```

2. Открыть раздел:

```text
Первый запуск
```

3. Нажать кнопку:

```text
Создать WSL venv и установить зависимости
```

Кнопка выполнит команду:

```bash
python scripts/setup_wsl_env.py --venv-dir .venv_wsl --requirements requirements.txt
```

Дальше скрипт внутри WSL выполнит:

```bash
python3 -m venv .venv_wsl
.venv_wsl/bin/python -m pip install --upgrade pip
.venv_wsl/bin/python -m pip install -r requirements.txt
```

## Через Windows bat

Можно запустить напрямую:

```bat
scripts\windows\setup_wsl_env.bat
```

## Через командную строку

```bash
python scripts/setup_wsl_env.py --venv-dir .venv_wsl --requirements requirements.txt
```

## Если в WSL не установлен python3-venv

Внутри WSL выполнить:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

После этого повторить установку зависимостей.

## Настройка в config

В `config/shelfvision.yaml` можно изменить путь к WSL-среде:

```yaml
setup:
  venv_dir: .venv
  venv_dir_wsl: .venv_wsl
  requirements: requirements.txt
```

`venv_dir` используется для Windows/local-запуска панели, а `venv_dir_wsl` — для установки зависимостей через WSL.
