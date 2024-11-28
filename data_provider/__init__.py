from .data_generator import (Dataset_C_MAPSS, Dataset_CWRU,
                             Dataset_Debutanizer, Dataset_SKAB, Dataset_SRU,
                             Dataset_SWaT, Dataset_TEP, Dataset_NASA_Li_ion)

DATA_DICT = {
    'SRU': Dataset_SRU,
    'CWRU': Dataset_CWRU,
    'C-MAPSS': Dataset_C_MAPSS,
    'SWaT': Dataset_SWaT,
    'Debutanizer': Dataset_Debutanizer,
    'TEP': Dataset_TEP,
    'SKAB': Dataset_SKAB,
    'NASA-Li-ion': Dataset_NASA_Li_ion
}
