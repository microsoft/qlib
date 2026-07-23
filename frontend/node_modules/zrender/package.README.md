# NOTICE about package.json

Only `zrender.js` are officially exported to users.

The other entries listed in the `"exports"` field of `package.json` make the internal files visible, but they are legacy usages, not recommended but not forbidden (for the sake of keeping backward compatible). These entries are made from the search in github about which internal files are imported.

Since `v5.5.0`, `"type": "module"` and `"exports: {...}"` are added to `package.json`. When upgrading to `v5.5.0+`, if you meet some problems about "can not find/resolve xxx" when importing some internal files, it probably because of the issue "file extension not fully specified". Please try to make the file extension fully specified (that is, `import 'xxx/xxx/xxx.js'` rather than `import 'xxx/xxx/xxx'`), or change the config of you bundler tools to support auto adding file extensions.

See more details about the `"exports"` field of `package.json` and why it is written like that in https://github.com/apache/echarts/pull/19513 .
