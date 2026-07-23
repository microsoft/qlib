
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
import { setAsPrimitive, map, isTypedArray, assert, each, retrieve2 } from 'zrender/lib/core/util.js';
import { createSource, cloneSourceShallow } from '../Source.js';
import { SOURCE_FORMAT_TYPED_ARRAY, SOURCE_FORMAT_ORIGINAL } from '../../util/types.js';
import { querySeriesUpstreamDatasetModel, queryDatasetUpstreamDatasetModels } from './sourceHelper.js';
import { applyDataTransform } from './transform.js';
import DataStore from '../DataStore.js';
import { DefaultDataProvider } from './dataProvider.js';
/**
 * [REQUIREMENT_MEMO]:
 * (0) `metaRawOption` means `dimensions`/`sourceHeader`/`seriesLayoutBy` in raw option.
 * (1) Keep support the feature: `metaRawOption` can be specified both on `series` and
 * `root-dataset`. Them on `series` has higher priority.
 * (2) Do not support to set `metaRawOption` on a `non-root-dataset`, because it might
 * confuse users: whether those props indicate how to visit the upstream source or visit
 * the transform result source, and some transforms has nothing to do with these props,
 * and some transforms might have multiple upstream.
 * (3) Transforms should specify `metaRawOption` in each output, just like they can be
 * declared in `root-dataset`.
 * (4) At present only support visit source in `SERIES_LAYOUT_BY_COLUMN` in transforms.
 * That is for reducing complexity in transforms.
 * PENDING: Whether to provide transposition transform?
 *
 * [IMPLEMENTAION_MEMO]:
 * "sourceVisitConfig" are calculated from `metaRawOption` and `data`.
 * They will not be calculated until `source` is about to be visited (to prevent from
 * duplicate calcuation). `source` is visited only in series and input to transforms.
 *
 * [DIMENSION_INHERIT_RULE]:
 * By default the dimensions are inherited from ancestors, unless a transform return
 * a new dimensions definition.
 * Consider the case:
 * ```js
 * dataset: [{
 *     source: [ ['Product', 'Sales', 'Prise'], ['Cookies', 321, 44.21], ...]
 * }, {
 *     transform: { type: 'filter', ... }
 * }]
 * dataset: [{
 *     dimension: ['Product', 'Sales', 'Prise'],
 *     source: [ ['Cookies', 321, 44.21], ...]
 * }, {
 *     transform: { type: 'filter', ... }
 * }]
 * ```
 * The two types of option should have the same behavior after transform.
 *
 *
 * [SCENARIO]:
 * (1) Provide source data directly:
 * ```js
 * series: {
 *     encode: {...},
 *     dimensions: [...]
 *     seriesLayoutBy: 'row',
 *     data: [[...]]
 * }
 * ```
 * (2) Series refer to dataset.
 * ```js
 * series: [{
 *     encode: {...}
 *     // Ignore datasetIndex means `datasetIndex: 0`
 *     // and the dimensions defination in dataset is used
 * }, {
 *     encode: {...},
 *     seriesLayoutBy: 'column',
 *     datasetIndex: 1
 * }]
 * ```
 * (3) dataset transform
 * ```js
 * dataset: [{
 *     source: [...]
 * }, {
 *     source: [...]
 * }, {
 *     // By default from 0.
 *     transform: { type: 'filter', config: {...} }
 * }, {
 *     // Piped.
 *     transform: [
 *         { type: 'filter', config: {...} },
 *         { type: 'sort', config: {...} }
 *     ]
 * }, {
 *     id: 'regressionData',
 *     fromDatasetIndex: 1,
 *     // Third-party transform
 *     transform: { type: 'ecStat:regression', config: {...} }
 * }, {
 *     // retrieve the extra result.
 *     id: 'regressionFormula',
 *     fromDatasetId: 'regressionData',
 *     fromTransformResult: 1
 * }]
 * ```
 */
var SourceManager = /** @class */function () {
  function SourceManager(sourceHost) {
    // Cached source. Do not repeat calculating if not dirty.
    this._sourceList = [];
    this._storeList = [];
    // version sign of each upstream source manager.
    this._upstreamSignList = [];
    this._versionSignBase = 0;
    this._dirty = true;
    this._sourceHost = sourceHost;
  }
  /**
   * Mark dirty.
   */
  SourceManager.prototype.dirty = function () {
    this._setLocalSource([], []);
    this._storeList = [];
    this._dirty = true;
  };
  SourceManager.prototype._setLocalSource = function (sourceList, upstreamSignList) {
    this._sourceList = sourceList;
    this._upstreamSignList = upstreamSignList;
    this._versionSignBase++;
    if (this._versionSignBase > 9e10) {
      this._versionSignBase = 0;
    }
  };
  /**
   * For detecting whether the upstream source is dirty, so that
   * the local cached source (in `_sourceList`) should be discarded.
   */
  SourceManager.prototype._getVersionSign = function () {
    return this._sourceHost.uid + '_' + this._versionSignBase;
  };
  /**
   * Always return a source instance. Otherwise throw error.
   */
  SourceManager.prototype.prepareSource = function () {
    // For the case that call `setOption` multiple time but no data changed,
    // cache the result source to prevent from repeating transform.
    if (this._isDirty()) {
      this._createSource();
      this._dirty = false;
    }
  };
  SourceManager.prototype._createSource = function () {
    this._setLocalSource([], []);
    var sourceHost = this._sourceHost;
    var upSourceMgrList = this._getUpstreamSourceManagers();
    var hasUpstream = !!upSourceMgrList.length;
    var resultSourceList;
    var upstreamSignList;
    if (isSeries(sourceHost)) {
      var seriesModel = sourceHost;
      var data = void 0;
      var sourceFormat = void 0;
      var upSource = void 0;
      // Has upstream dataset
      if (hasUpstream) {
        var upSourceMgr = upSourceMgrList[0];
        upSourceMgr.prepareSource();
        upSource = upSourceMgr.getSource();
        data = upSource.data;
        sourceFormat = upSource.sourceFormat;
        upstreamSignList = [upSourceMgr._getVersionSign()];
      }
      // Series data is from own.
      else {
        data = seriesModel.get('data', true);
        sourceFormat = isTypedArray(data) ? SOURCE_FORMAT_TYPED_ARRAY : SOURCE_FORMAT_ORIGINAL;
        upstreamSignList = [];
      }
      // See [REQUIREMENT_MEMO], merge settings on series and parent dataset if it is root.
      var newMetaRawOption = this._getSourceMetaRawOption() || {};
      var upMetaRawOption = upSource && upSource.metaRawOption || {};
      var seriesLayoutBy = retrieve2(newMetaRawOption.seriesLayoutBy, upMetaRawOption.seriesLayoutBy) || null;
      var sourceHeader = retrieve2(newMetaRawOption.sourceHeader, upMetaRawOption.sourceHeader);
      // Note here we should not use `upSource.dimensionsDefine`. Consider the case:
      // `upSource.dimensionsDefine` is detected by `seriesLayoutBy: 'column'`,
      // but series need `seriesLayoutBy: 'row'`.
      var dimensions = retrieve2(newMetaRawOption.dimensions, upMetaRawOption.dimensions);
      // We share source with dataset as much as possible
      // to avoid extra memory cost of high dimensional data.
      var needsCreateSource = seriesLayoutBy !== upMetaRawOption.seriesLayoutBy || !!sourceHeader !== !!upMetaRawOption.sourceHeader || dimensions;
      resultSourceList = needsCreateSource ? [createSource(data, {
        seriesLayoutBy: seriesLayoutBy,
        sourceHeader: sourceHeader,
        dimensions: dimensions
      }, sourceFormat)] : [];
    } else {
      var datasetModel = sourceHost;
      // Has upstream dataset.
      if (hasUpstream) {
        var result = this._applyTransform(upSourceMgrList);
        resultSourceList = result.sourceList;
        upstreamSignList = result.upstreamSignList;
      }
      // Is root dataset.
      else {
        var sourceData = datasetModel.get('source', true);
        resultSourceList = [createSource(sourceData, this._getSourceMetaRawOption(), null)];
        upstreamSignList = [];
      }
    }
    if (process.env.NODE_ENV !== 'production') {
      assert(resultSourceList && upstreamSignList);
    }
    this._setLocalSource(resultSourceList, upstreamSignList);
  };
  SourceManager.prototype._applyTransform = function (upMgrList) {
    var datasetModel = this._sourceHost;
    var transformOption = datasetModel.get('transform', true);
    var fromTransformResult = datasetModel.get('fromTransformResult', true);
    if (process.env.NODE_ENV !== 'production') {
      assert(fromTransformResult != null || transformOption != null);
    }
    if (fromTransformResult != null) {
      var errMsg = '';
      if (upMgrList.length !== 1) {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = 'When using `fromTransformResult`, there should be only one upstream dataset';
        }
        doThrow(errMsg);
      }
    }
    var sourceList;
    var upSourceList = [];
    var upstreamSignList = [];
    each(upMgrList, function (upMgr) {
      upMgr.prepareSource();
      var upSource = upMgr.getSource(fromTransformResult || 0);
      var errMsg = '';
      if (fromTransformResult != null && !upSource) {
        if (process.env.NODE_ENV !== 'production') {
          errMsg = 'Can not retrieve result by `fromTransformResult`: ' + fromTransformResult;
        }
        doThrow(errMsg);
      }
      upSourceList.push(upSource);
      upstreamSignList.push(upMgr._getVersionSign());
    });
    if (transformOption) {
      sourceList = applyDataTransform(transformOption, upSourceList, {
        datasetIndex: datasetModel.componentIndex
      });
    } else if (fromTransformResult != null) {
      sourceList = [cloneSourceShallow(upSourceList[0])];
    }
    return {
      sourceList: sourceList,
      upstreamSignList: upstreamSignList
    };
  };
  SourceManager.prototype._isDirty = function () {
    if (this._dirty) {
      return true;
    }
    // All sourceList is from the some upstream.
    var upSourceMgrList = this._getUpstreamSourceManagers();
    for (var i = 0; i < upSourceMgrList.length; i++) {
      var upSrcMgr = upSourceMgrList[i];
      if (
      // Consider the case that there is ancestor diry, call it recursively.
      // The performance is probably not an issue because usually the chain is not long.
      upSrcMgr._isDirty() || this._upstreamSignList[i] !== upSrcMgr._getVersionSign()) {
        return true;
      }
    }
  };
  /**
   * @param sourceIndex By default 0, means "main source".
   *                    In most cases there is only one source.
   */
  SourceManager.prototype.getSource = function (sourceIndex) {
    sourceIndex = sourceIndex || 0;
    var source = this._sourceList[sourceIndex];
    if (!source) {
      // Series may share source instance with dataset.
      var upSourceMgrList = this._getUpstreamSourceManagers();
      return upSourceMgrList[0] && upSourceMgrList[0].getSource(sourceIndex);
    }
    return source;
  };
  /**
   *
   * Get a data store which can be shared across series.
   * Only available for series.
   *
   * @param seriesDimRequest Dimensions that are generated in series.
   *        Should have been sorted by `storeDimIndex` asc.
   */
  SourceManager.prototype.getSharedDataStore = function (seriesDimRequest) {
    if (process.env.NODE_ENV !== 'production') {
      assert(isSeries(this._sourceHost), 'Can only call getDataStore on series source manager.');
    }
    var schema = seriesDimRequest.makeStoreSchema();
    return this._innerGetDataStore(schema.dimensions, seriesDimRequest.source, schema.hash);
  };
  SourceManager.prototype._innerGetDataStore = function (storeDims, seriesSource, sourceReadKey) {
    // TODO Can use other sourceIndex?
    var sourceIndex = 0;
    var storeList = this._storeList;
    var cachedStoreMap = storeList[sourceIndex];
    if (!cachedStoreMap) {
      cachedStoreMap = storeList[sourceIndex] = {};
    }
    var cachedStore = cachedStoreMap[sourceReadKey];
    if (!cachedStore) {
      var upSourceMgr = this._getUpstreamSourceManagers()[0];
      if (isSeries(this._sourceHost) && upSourceMgr) {
        cachedStore = upSourceMgr._innerGetDataStore(storeDims, seriesSource, sourceReadKey);
      } else {
        cachedStore = new DataStore();
        // Always create store from source of series.
        cachedStore.initData(new DefaultDataProvider(seriesSource, storeDims.length), storeDims);
      }
      cachedStoreMap[sourceReadKey] = cachedStore;
    }
    return cachedStore;
  };
  /**
   * PENDING: Is it fast enough?
   * If no upstream, return empty array.
   */
  SourceManager.prototype._getUpstreamSourceManagers = function () {
    // Always get the relationship from the raw option.
    // Do not cache the link of the dependency graph, so that
    // there is no need to update them when change happens.
    var sourceHost = this._sourceHost;
    if (isSeries(sourceHost)) {
      var datasetModel = querySeriesUpstreamDatasetModel(sourceHost);
      return !datasetModel ? [] : [datasetModel.getSourceManager()];
    } else {
      return map(queryDatasetUpstreamDatasetModels(sourceHost), function (datasetModel) {
        return datasetModel.getSourceManager();
      });
    }
  };
  SourceManager.prototype._getSourceMetaRawOption = function () {
    var sourceHost = this._sourceHost;
    var seriesLayoutBy;
    var sourceHeader;
    var dimensions;
    if (isSeries(sourceHost)) {
      seriesLayoutBy = sourceHost.get('seriesLayoutBy', true);
      sourceHeader = sourceHost.get('sourceHeader', true);
      dimensions = sourceHost.get('dimensions', true);
    }
    // See [REQUIREMENT_MEMO], `non-root-dataset` do not support them.
    else if (!this._getUpstreamSourceManagers().length) {
      var model = sourceHost;
      seriesLayoutBy = model.get('seriesLayoutBy', true);
      sourceHeader = model.get('sourceHeader', true);
      dimensions = model.get('dimensions', true);
    }
    return {
      seriesLayoutBy: seriesLayoutBy,
      sourceHeader: sourceHeader,
      dimensions: dimensions
    };
  };
  return SourceManager;
}();
export { SourceManager };
// Call this method after `super.init` and `super.mergeOption` to
// disable the transform merge, but do not disable transform clone from rawOption.
export function disableTransformOptionMerge(datasetModel) {
  var transformOption = datasetModel.option.transform;
  transformOption && setAsPrimitive(datasetModel.option.transform);
}
function isSeries(sourceHost) {
  // Avoid circular dependency with Series.ts
  return sourceHost.mainType === 'series';
}
function doThrow(errMsg) {
  throw new Error(errMsg);
}