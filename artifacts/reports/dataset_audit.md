# SROIE Dataset Audit

- Valid image/entity pairs: **973**
- Train pairs: **626**
- Test pairs: **347**
- Excluded samples: **0**

## Canonical field coverage

| Field | Non-empty | Empty |
|---|---:|---:|
| company | 973 | 0 |
| address | 972 | 1 |
| date | 973 | 0 |
| total | 972 | 1 |

## Image dimensions

- Width: {'min': 435, 'max': 4961, 'mean': 1359.56, 'median': 825}
- Height: {'min': 605, 'max': 7016, 'mean': 2401.24, 'median': 1697}

## Exclusions

`{}`

Targets preserve strings and serialize in the stable order `company`, `address`, `date`, `total`. Missing values use an empty string.
