# Crowd Source Data

## Initiative
Public data source like yahoo is flawed, it might miss data for stock which is delisted and it might have data which is wrong. This can introduce survivorship bias into our training process.

The Crowd Source Data is introduced to merged data from multiple data source and cross validate against each other, so that:
1. We will have a more complete history record.
2. We can identify the anomaly data and apply correction when necessary.

## Related Repo
The raw data is hosted on dolthub repo: https://www.dolthub.com/repositories/chenditc/investment_data

The processing script and sql is hosted on github repo: https://github.com/chenditc/investment_data

The packaged docker runtime is hosted on dockerhub: https://hub.docker.com/repository/docker/chenditc/investment_data

## How to use it in qlib
### Option 1: Download release bin data
User can download data in qlib bin format and use it directly: https://github.com/chenditc/investment_data/releases/tag/20220720
```bash
wget https://github.com/chenditc/investment_data/releases/download/20220720/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

### Option 2: Generate qlib data from dolthub
Dolthub data will be update daily, so that if user wants to get up to date data, they can dump qlib bin using docker:
```
docker run -v /<some output directory>:/output -it --rm chenditc/investment_data bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

## FAQ and other info
See: https://github.com/chenditc/investment_data/blob/main/README.md
