# Data

## The real ESG dataset is NOT included

The original notebooks read a local file called `PaperReady_numeric_esg.txt` (495 companies).
That file was not available when this repository was assembled, and ESG ratings usually come
from licensed providers, so nothing here recreates or invents it.

To run the full experiment, put your own copy at `data/raw/PaperReady_numeric_esg.txt`
(`data/raw/` is git-ignored, so the data never gets pushed by accident).

## Accepted formats

**1. Notebook format (text file with a Python dict):**

```python
{
    "AAPL": [72.1, 73.4, 71.9, ...],
    "MSFT": [80.2, 81.0, ...],
}
```

Values are the company's ESG scores in time order (oldest first). Missing values (`None` / `nan`) are dropped.

**2. CSV with three columns:**

```text
ticker,date,esg_score
AAPL,2016-12-31,72.1
AAPL,2017-12-31,73.4
```

Rows are sorted by date for each ticker automatically.

## Sample data (synthetic)

`sample/synthetic_esg_demo.txt` contains 30 **fake** series (bounded random walks) so the code and tests can run.
It has no connection to real companies or real ESG scores.
