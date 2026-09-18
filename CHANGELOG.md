# Changelog

## [0.9.8](https://github.com/microsoft/qlib/compare/v0.9.7...v0.9.8) (2025-11-13)


### Bug Fixes

* download orderbook data error ([#1990](https://github.com/microsoft/qlib/issues/1990)) ([136b2dd](https://github.com/microsoft/qlib/commit/136b2ddf9a16e4106d62b8d1336a56273a8abef0))
* **gbdt:** correct dtrain assignment in finetune() to use Dataset instead of tuple ([#2049](https://github.com/microsoft/qlib/issues/2049)) ([2b41782](https://github.com/microsoft/qlib/commit/2b41782f0cfb81e8cc065f2915b215758a7838ef))
* **macd:** remove extra division by close in DEA calculation to ensure dimension consistency ([#2046](https://github.com/microsoft/qlib/issues/2046)) ([66c3622](https://github.com/microsoft/qlib/commit/66c36226aafceabe497e5967f67921e5d3c9d497))
* replace deprecated pandas fillna(method=) with ffill()/bfill() ([#1987](https://github.com/microsoft/qlib/issues/1987)) ([7095e75](https://github.com/microsoft/qlib/commit/7095e755fa57e011f0483d24b45fc5bd5a4deaf8))
* spelling errors ([#1996](https://github.com/microsoft/qlib/issues/1996)) ([f26b341](https://github.com/microsoft/qlib/commit/f26b3417363410531dbbb39e425bce6cf05528a1))
* the bug when auto_mount=True ([#2009](https://github.com/microsoft/qlib/issues/2009)) ([213eb6c](https://github.com/microsoft/qlib/commit/213eb6c2cd12342b6ec98f21300217e1659f3d58))
* typo in integration documentation: 'userd' -&gt; 'used' ([#2034](https://github.com/microsoft/qlib/issues/2034)) ([3dc5a7d](https://github.com/microsoft/qlib/commit/3dc5a7d299074f0fa45a4b7bb50ab446a8824a32))
