# iBOVESPA History Companies Collection

## Requirements

- Install the libs from the file `requirements.txt`

    ```bash
    pip install -r requirements.txt
    ```
- `requirements.txt` file was generated using python3.8

## For the ibovespa (IBOV) index, we have:

<hr/>

### Method `get_new_companies`

#### <b>Index start date</b>

- The ibovespa index started on 2 January 1968 ([wiki](https://en.wikipedia.org/wiki/%C3%8Dndice_Bovespa)).  In order to use this start date in our `bench_start_date(self)` method, two conditions must be satisfied:
    1) APIs used to download brazilian stocks (B3) historical prices must keep track of such historic data since 2 January 1968

    2) Some website or API must provide, from that date, the historic index composition. In other words, the companies used to build the index .

    As a consequence, the method `bench_start_date(self)` inside `collector.py` was implemented using `pd.Timestamp("2003-01-03")` due to two reasons

    1) The earliest ibov composition that have been found was from the first quarter of 2003. More informations about such composition can be seen on the sections below.

    2) Yahoo finance, one of the libraries used to download symbols historic prices, keeps track from this date forward.

- Within the `get_new_companies` method, a logic was implemented to get, for each ibovespa component stock, the start date that yahoo finance keeps track of.

#### <b>Code Logic</b>

The code does a web scrapping into the B3's [website](https://sistemaswebb3-listados.b3.com.br/indexPage/day/IBOV?language=pt-br), which keeps track of the ibovespa stocks composition on the current day. 

Other approaches, such as `request` and `Beautiful Soup` could have been used. However, the website shows the table with the stocks with some delay, since it uses a script inside of it to obtain such compositions.
Alternatively, `selenium` was used to download this stocks' composition in order to overcome this problem.

Futhermore, the data downloaded from the selenium script  was preprocessed so it could be saved into the `csv` format stablished by `scripts/data_collector/index.py`.

<hr/>

### Method `get_changes` 

No suitable data source that keeps track of ibovespa's history stocks composition has been found. Except from this [repository](https://github.com/igor17400/IBOV-HCI) which provide such information have been used, however it only provides the data from the 1st quarter of 2003 to 3rd quarter of 2021.

With that reference, the index's composition can be compared quarter by quarter and year by year and then generate a file that keeps track of which stocks have been removed and which have been added each quarter and year.

<hr/>

### Collector Data

```bash
# parse instruments, using in qlib/instruments.
python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method parse_instruments

# parse new companies
python collector.py --index_name IBOV --qlib_dir ~/.qlib/qlib_data/br_data --method save_new_companies
```

