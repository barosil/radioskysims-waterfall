# radioskysims-waterfall

Ferramentas de gerenciamento de **waterfalls** (tempo × frequência) para
radioastronomia, com foco em processamento seguro de grandes volumes de dados
(xarray + dask), arquitetura hexagonal (domain / application / ports /
adapters) e visualização preservando memória (holoviews + datashader).

## Destaques

- **Value object `Waterfall`** — array 2D (tempo × frequência) + metadados,
  com métodos *chainable* (slice, resample, filtros, SED, lightcurve, ...).
- **Carregamento lazy** via netcdf, hdf5, zarr e fits, com opções de
  mapeamento de nomes (`freq_name`, `time_name`, `data_name`).
- **Ferramentas** de slicing, resampling, decimation, interpolação, rebinning,
  conversão de unidades, máscaras, eventos, filtros (mean/median/MAD/ArPLS),
  bandpass, baseline, SED, lightcurve e normalização.
- **Pipeline** declarativo e *chainable*, com verificação segura de CPU/memória.
- **Persistência** com controle de versão de arquivos.
- **Visualização** de um ou dois waterfalls lado a lado (holoviews + datashader).

## Instalação

```bash
pip install -e "packages/radioskysims-waterfall[all]"
# ou seletivo:
pip install -e "packages/radioskysims-waterfall[hdf5,plot]"
```

## Exemplo mínimo

```python
from radioskysims_waterfall import Waterfall
import xarray as xr
import numpy as np

wf = Waterfall(xr.DataArray(
    np.random.rand(100, 64),
    dims=("time", "freq"),
    coords={"time": np.arange(100), "freq": np.arange(64)},
))

# Chainable
result = (
    wf.resample(freq_factor=2)
      .filter_spectral(method="median", window=5)
      .get_bandpass(threshold=3.0, sigmas=1.5)
)
sed = wf.sed(interval=slice(0, 50))
lc = wf.lightcurve(frequencies=slice(10, 40))
```

## Layout (hexagonal)

```
src/radioskysims_waterfall/
├── domain/        # modelos, operações, filtros, SED, lightcurve, baseline...
├── application/   # ports (contratos), serviços (pipeline)
├── adapters/      # loaders (netcdf/hdf5/zarr/fits), storage
└── infrastructure/# monitor de recursos (CPU/memória)
```
