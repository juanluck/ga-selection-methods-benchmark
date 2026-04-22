# Reproducción del paper BMT en Python (lista para GitHub)

Este repositorio contiene una reproducción práctica del paper de **Bipolar Mating Tendency (BMT)** centrada en:
- los **21 benchmarks clásicos**
- el bloque **CEC 2017 bound-constrained** (el que en muchos trabajos se usa como referencia del bloque bound-constrained de 2018)
- los **3 problemas de ingeniería** del paper
- el barrido de **bipolarity**
- las figuras de **diversidad** y **footprints**

El objetivo es que puedas subirlo directamente a GitHub y ejecutarlo en un entorno Linux tipo Debian con el menor número posible de pasos manuales.

## Qué incluye

Código principal:
- `src/bmt_repro/`: implementación del GA, operadores de selección, benchmarks y utilidades estadísticas.
- `run_basic_benchmarks.py`: 21 benchmarks clásicos.
- `run_cec2017_bound_benchmarks.py`: bloque CEC usando `cec2017-py`.
- `run_engineering_problems.py`: problemas de ingeniería.
- `run_bipolarity_sweep.py`: barrido de `q` en BMT.
- `run_diversity_figures.py`: figuras tipo paper.
- `scripts/smoke_test.py`: prueba rápida para verificar que todo está bien instalado.

Se han eliminado archivos superfluos de versiones anteriores:
- proxies antiguos de CEC 2018
- plantillas antiguas para el bundle oficial
- archivos auxiliares que no hacen falta para subir este proyecto a GitHub

## Métodos de selección incluidos

Los seis métodos comparados en el paper:
- `ST`
- `RTS`
- `UTS`
- `FGTS`
- `CS`
- `BMT`

## Requisitos del sistema (Debian / Ubuntu)

Instala primero Python y las dependencias del sistema:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential
```

También puedes usar el script incluido:

```bash
bash scripts/install_debian.sh
```

## Crear el entorno virtual e instalar dependencias

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

o, si prefieres, con el script:

```bash
bash scripts/setup_venv.sh
```

El fichero `requirements.txt` ya incluye la instalación de `cec2017-py` desde GitHub.  
Si quieres instalarlo manualmente, el comando es:

```bash
pip install git+https://github.com/tilleyd/cec2017-py.git
```

## Instalación opcional del paquete en modo editable

No es obligatorio, pero puedes dejar el proyecto instalado en editable:

```bash
pip install -e .
```

## Prueba rápida

Para comprobar que el entorno funciona:

```bash
python scripts/smoke_test.py
```

Esa prueba ejecuta:
- `dimension = 10`
- `runs = 1`
- `population_size = 100`
- `evals_per_dim = 100`
- algoritmos `ST BMT CS`

## Cómo ejecutar experimentos

### 1) 21 benchmarks clásicos

```bash
python run_basic_benchmarks.py
```

Ejemplo con solo `ST`, `BMT` y `CS`:

```bash
python run_basic_benchmarks.py --algorithms ST BMT CS
```

Ejemplo corto:

```bash
python run_basic_benchmarks.py --runs 3 --population-sizes 100 --algorithms ST BMT CS
```

### 2) CEC 2017 bound-constrained

```bash
python run_cec2017_bound_benchmarks.py
```

Ejemplo corto en 10 dimensiones, una sola población y un solo run:

```bash
python run_cec2017_bound_benchmarks.py \
  --runs 1 \
  --population-sizes 100 \
  --dimension 10 \
  --evals-per-dim 100 \
  --algorithms ST BMT CS
```

Notas:
- por defecto usa `dimension = 10`
- por defecto excluye `F2`, igual que hacen muchas implementaciones modernas del bloque
- si no fijas `--generations`, el script calcula generaciones a partir de `evals_per_dim * dimension`

### 3) Problemas de ingeniería

```bash
python run_engineering_problems.py
```

### 4) Barrido de bipolarity

```bash
python run_bipolarity_sweep.py --population-size 100
```

### 5) Figuras de diversidad

```bash
python run_diversity_figures.py
```

## Salidas

Todos los scripts guardan resultados en `results/...`.

Los archivos principales suelen ser:
- `raw_results.csv`
- `summary_median_std.csv`
- `friedman_by_population.csv`
- `wilcoxon_by_population.csv`

En el caso del runner CEC también se genera:
- `run_metadata.csv`

## Parámetros tipo-paper usados por defecto

- `pc = 0.7`
- `pm = 0.05`
- `lambda = 0.6`
- `tournament_size = 4`
- `population_sizes = 50, 100, 200`
- `runs = 25`
- `bipolarity = 0.25`

Parámetros reconstruidos y editables:
- `fgts_ftour = 4.5`
- `rts_window = 4`
- `association_size = 4`

## Reproducibilidad

Se ha corregido la generación de semillas para que sea **estable entre ejecuciones y máquinas**.  
Ya no depende del `hash()` aleatorio de Python.

## Estructura del repositorio

```text
.
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_basic_benchmarks.py
├── run_bipolarity_sweep.py
├── run_cec2017_bound_benchmarks.py
├── run_diversity_figures.py
├── run_engineering_problems.py
├── scripts
│   ├── install_debian.sh
│   ├── setup_venv.sh
│   └── smoke_test.py
└── src
    └── bmt_repro
        ├── __init__.py
        ├── benchmarks.py
        ├── cec2017_suite.py
        ├── diversity.py
        ├── engineering.py
        ├── ga.py
        ├── selection.py
        ├── stats.py
        └── utils.py
```

## Subirlo a GitHub

Ejemplo mínimo:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin TU_URL_DE_GITHUB
git push -u origin main
```

## Aviso

Este proyecto es una **reconstrucción práctica en Python** del setup del paper.  
No afirma ser una reproducción byte a byte del código MATLAB original de los autores.
