
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
/* global Int32Array */
import * as zrUtil from 'zrender/lib/core/util.js';
import Model from '../model/Model.js';
import DataDiffer from './DataDiffer.js';
import { DefaultDataProvider } from './helper/dataProvider.js';
import { summarizeDimensions } from './helper/dimensionHelper.js';
import SeriesDimensionDefine from './SeriesDimensionDefine.js';
import { SOURCE_FORMAT_TYPED_ARRAY, SOURCE_FORMAT_ORIGINAL } from '../util/types.js';
import { convertOptionIdName, isDataItemOption } from '../util/model.js';
import { setCommonECData } from '../util/innerStore.js';
import { isSourceInstance } from './Source.js';
import DataStore from './DataStore.js';
import { isSeriesDataSchema } from './helper/SeriesDataSchema.js';
var isObject = zrUtil.isObject;
var map = zrUtil.map;
var CtorInt32Array = typeof Int32Array === 'undefined' ? Array : Int32Array;
// Use prefix to avoid index to be the same as otherIdList[idx],
// which will cause weird update animation.
var ID_PREFIX = 'e\0\0';
var INDEX_NOT_FOUND = -1;
// type SeriesDimensionIndex = DimensionIndex;
var TRANSFERABLE_PROPERTIES = ['hasItemOption', '_nameList', '_idList', '_invertedIndicesMap', '_dimSummary', 'userOutput', '_rawData', '_dimValueGetter', '_nameDimIdx', '_idDimIdx', '_nameRepeatCount'];
var CLONE_PROPERTIES = ['_approximateExtent'];
// -----------------------------
// Internal method declarations:
// -----------------------------
var prepareInvertedIndex;
var getId;
var getIdNameFromStore;
var normalizeDimensions;
var transferProperties;
var cloneListForMapAndSample;
var makeIdFromName;
var SeriesData = /** @class */function () {
  /**
   * @param dimensionsInput.dimensions
   *        For example, ['someDimName', {name: 'someDimName', type: 'someDimType'}, ...].
   *        Dimensions should be concrete names like x, y, z, lng, lat, angle, radius
   */
  function SeriesData(dimensionsInput, hostModel) {
    this.type = 'list';
    this._dimOmitted = false;
    this._nameList = [];
    this._idList = [];
    // Models of data option is stored sparse for optimizing memory cost
    // Never used yet (not used yet).
    // private _optionModels: Model[] = [];
    // Global visual properties after visual coding
    this._visual = {};
    // Global layout properties.
    this._layout = {};
    // Item visual properties after visual coding
    this._itemVisuals = [];
    // Item layout properties after layout
    this._itemLayouts = [];
    // Graphic elements
    this._graphicEls = [];
    // key: dim, value: extent
    this._approximateExtent = {};
    this._calculationInfo = {};
    // Having detected that there is data item is non primitive type
    // (in type `OptionDataItemObject`).
    // Like `data: [ { value: xx, itemStyle: {...} }, ...]`
    // At present it only happen in `SOURCE_FORMAT_ORIGINAL`.
    this.hasItemOption = false;
    // Methods that create a new list based on this list should be listed here.
    // Notice that those method should `RETURN` the new list.
    this.TRANSFERABLE_METHODS = ['cloneShallow', 'downSample', 'minmaxDownSample', 'lttbDownSample', 'map'];
    // Methods that change indices of this list should be listed here.
    this.CHANGABLE_METHODS = ['filterSelf', 'selectRange'];
    this.DOWNSAMPLE_METHODS = ['downSample', 'minmaxDownSample', 'lttbDownSample'];
    var dimensions;
    var assignStoreDimIdx = false;
    if (isSeriesDataSchema(dimensionsInput)) {
      dimensions = dimensionsInput.dimensions;
      this._dimOmitted = dimensionsInput.isDimensionOmitted();
      this._schema = dimensionsInput;
    } else {
      assignStoreDimIdx = true;
      dimensions = dimensionsInput;
    }
    dimensions = dimensions || ['x', 'y'];
    var dimensionInfos = {};
    var dimensionNames = [];
    var invertedIndicesMap = {};
    var needsHasOwn = false;
    var emptyObj = {};
    for (var i = 0; i < dimensions.length; i++) {
      // Use the original dimensions[i], where other flag props may exists.
      var dimInfoInput = dimensions[i];
      var dimensionInfo = zrUtil.isString(dimInfoInput) ? new SeriesDimensionDefine({
        name: dimInfoInput
      }) : !(dimInfoInput instanceof SeriesDimensionDefine) ? new SeriesDimensionDefine(dimInfoInput) : dimInfoInput;
      var dimensionName = dimensionInfo.name;
      dimensionInfo.type = dimensionInfo.type || 'float';
      if (!dimensionInfo.coordDim) {
        dimensionInfo.coordDim = dimensionName;
        dimensionInfo.coordDimIndex = 0;
      }
      var otherDims = dimensionInfo.otherDims = dimensionInfo.otherDims || {};
      dimensionNames.push(dimensionName);
      dimensionInfos[dimensionName] = dimensionInfo;
      if (emptyObj[dimensionName] != null) {
        needsHasOwn = true;
      }
      if (dimensionInfo.createInvertedIndices) {
        invertedIndicesMap[dimensionName] = [];
      }
      if (otherDims.itemName === 0) {
        this._nameDimIdx = i;
      }
      if (otherDims.itemId === 0) {
        this._idDimIdx = i;
      }
      if (process.env.NODE_ENV !== 'production') {
        zrUtil.assert(assignStoreDimIdx || dimensionInfo.storeDimIndex >= 0);
      }
      if (assignStoreDimIdx) {
        dimensionInfo.storeDimIndex = i;
      }
    }
    this.dimensions = dimensionNames;
    this._dimInfos = dimensionInfos;
    this._initGetDimensionInfo(needsHasOwn);
    this.hostModel = hostModel;
    this._invertedIndicesMap = invertedIndicesMap;
    if (this._dimOmitted) {
      var dimIdxToName_1 = this._dimIdxToName = zrUtil.createHashMap();
      zrUtil.each(dimensionNames, function (dimName) {
        dimIdxToName_1.set(dimensionInfos[dimName].storeDimIndex, dimName);
      });
    }
  }
  /**
   *
   * Get concrete dimension name by dimension name or dimension index.
   * If input a dimension name, do not validate whether the dimension name exits.
   *
   * @caution
   * @param dim Must make sure the dimension is `SeriesDimensionLoose`.
   * Because only those dimensions will have auto-generated dimension names if not
   * have a user-specified name, and other dimensions will get a return of null/undefined.
   *
   * @notice Because of this reason, should better use `getDimensionIndex` instead, for examples:
   * ```js
   * const val = data.getStore().get(data.getDimensionIndex(dim), dataIdx);
   * ```
   *
   * @return Concrete dim name.
   */
  SeriesData.prototype.getDimension = function (dim) {
    var dimIdx = this._recognizeDimIndex(dim);
    if (dimIdx == null) {
      return dim;
    }
    dimIdx = dim;
    if (!this._dimOmitted) {
      return this.dimensions[dimIdx];
    }
    // Retrieve from series dimension definition because it probably contains
    // generated dimension name (like 'x', 'y').
    var dimName = this._dimIdxToName.get(dimIdx);
    if (dimName != null) {
      return dimName;
    }
    var sourceDimDef = this._schema.getSourceDimension(dimIdx);
    if (sourceDimDef) {
      return sourceDimDef.name;
    }
  };
  /**
   * Get dimension index in data store. Return -1 if not found.
   * Can be used to index value from getRawValue.
   */
  SeriesData.prototype.getDimensionIndex = function (dim) {
    var dimIdx = this._recognizeDimIndex(dim);
    if (dimIdx != null) {
      return dimIdx;
    }
    if (dim == null) {
      return -1;
    }
    var dimInfo = this._getDimInfo(dim);
    return dimInfo ? dimInfo.storeDimIndex : this._dimOmitted ? this._schema.getSourceDimensionIndex(dim) : -1;
  };
  /**
   * The meanings of the input parameter `dim`:
   *
   * + If dim is a number (e.g., `1`), it means the index of the dimension.
   *   For example, `getDimension(0)` will return 'x' or 'lng' or 'radius'.
   * + If dim is a number-like string (e.g., `"1"`):
   *     + If there is the same concrete dim name defined in `series.dimensions` or `dataset.dimensions`,
   *        it means that concrete name.
   *     + If not, it will be converted to a number, which means the index of the dimension.
   *        (why? because of the backward compatibility. We have been tolerating number-like string in
   *        dimension setting, although now it seems that it is not a good idea.)
   *     For example, `visualMap[i].dimension: "1"` is the same meaning as `visualMap[i].dimension: 1`,
   *     if no dimension name is defined as `"1"`.
   * + If dim is a not-number-like string, it means the concrete dim name.
   *   For example, it can be be default name `"x"`, `"y"`, `"z"`, `"lng"`, `"lat"`, `"angle"`, `"radius"`,
   *   or customized in `dimensions` property of option like `"age"`.
   *
   * @return recognized `DimensionIndex`. Otherwise return null/undefined (means that dim is `DimensionName`).
   */
  SeriesData.prototype._recognizeDimIndex = function (dim) {
    if (zrUtil.isNumber(dim)
    // If being a number-like string but not being defined as a dimension name.
    || dim != null && !isNaN(dim) && !this._getDimInfo(dim) && (!this._dimOmitted || this._schema.getSourceDimensionIndex(dim) < 0)) {
      return +dim;
    }
  };
  SeriesData.prototype._getStoreDimIndex = function (dim) {
    var dimIdx = this.getDimensionIndex(dim);
    if (process.env.NODE_ENV !== 'production') {
      if (dimIdx == null) {
        throw new Error('Unknown dimension ' + dim);
      }
    }
    return dimIdx;
  };
  /**
   * Get type and calculation info of particular dimension
   * @param dim
   *        Dimension can be concrete names like x, y, z, lng, lat, angle, radius
   *        Or a ordinal number. For example getDimensionInfo(0) will return 'x' or 'lng' or 'radius'
   */
  SeriesData.prototype.getDimensionInfo = function (dim) {
    // Do not clone, because there may be categories in dimInfo.
    return this._getDimInfo(this.getDimension(dim));
  };
  SeriesData.prototype._initGetDimensionInfo = function (needsHasOwn) {
    var dimensionInfos = this._dimInfos;
    this._getDimInfo = needsHasOwn ? function (dimName) {
      return dimensionInfos.hasOwnProperty(dimName) ? dimensionInfos[dimName] : undefined;
    } : function (dimName) {
      return dimensionInfos[dimName];
    };
  };
  /**
   * concrete dimension name list on coord.
   */
  SeriesData.prototype.getDimensionsOnCoord = function () {
    return this._dimSummary.dataDimsOnCoord.slice();
  };
  SeriesData.prototype.mapDimension = function (coordDim, idx) {
    var dimensionsSummary = this._dimSummary;
    if (idx == null) {
      return dimensionsSummary.encodeFirstDimNotExtra[coordDim];
    }
    var dims = dimensionsSummary.encode[coordDim];
    return dims ? dims[idx] : null;
  };
  SeriesData.prototype.mapDimensionsAll = function (coordDim) {
    var dimensionsSummary = this._dimSummary;
    var dims = dimensionsSummary.encode[coordDim];
    return (dims || []).slice();
  };
  SeriesData.prototype.getStore = function () {
    return this._store;
  };
  /**
   * Initialize from data
   * @param data source or data or data store.
   * @param nameList The name of a datum is used on data diff and
   *        default label/tooltip.
   *        A name can be specified in encode.itemName,
   *        or dataItem.name (only for series option data),
   *        or provided in nameList from outside.
   */
  SeriesData.prototype.initData = function (data, nameList, dimValueGetter) {
    var _this = this;
    var store;
    if (data instanceof DataStore) {
      store = data;
    }
    if (!store) {
      var dimensions = this.dimensions;
      var provider = isSourceInstance(data) || zrUtil.isArrayLike(data) ? new DefaultDataProvider(data, dimensions.length) : data;
      store = new DataStore();
      var dimensionInfos = map(dimensions, function (dimName) {
        return {
          type: _this._dimInfos[dimName].type,
          property: dimName
        };
      });
      store.initData(provider, dimensionInfos, dimValueGetter);
    }
    this._store = store;
    // Reset
    this._nameList = (nameList || []).slice();
    this._idList = [];
    this._nameRepeatCount = {};
    this._doInit(0, store.count());
    // Cache summary info for fast visit. See "dimensionHelper".
    // Needs to be initialized after store is prepared.
    this._dimSummary = summarizeDimensions(this, this._schema);
    this.userOutput = this._dimSummary.userOutput;
  };
  /**
   * Caution: Can be only called on raw data (before `this._indices` created).
   */
  SeriesData.prototype.appendData = function (data) {
    var range = this._store.appendData(data);
    this._doInit(range[0], range[1]);
  };
  /**
   * Caution: Can be only called on raw data (before `this._indices` created).
   * This method does not modify `rawData` (`dataProvider`), but only
   * add values to store.
   *
   * The final count will be increased by `Math.max(values.length, names.length)`.
   *
   * @param values That is the SourceType: 'arrayRows', like
   *        [
   *            [12, 33, 44],
   *            [NaN, 43, 1],
   *            ['-', 'asdf', 0]
   *        ]
   *        Each item is exactly corresponding to a dimension.
   */
  SeriesData.prototype.appendValues = function (values, names) {
    var _a = this._store.appendValues(values, names && names.length),
      start = _a.start,
      end = _a.end;
    var shouldMakeIdFromName = this._shouldMakeIdFromName();
    this._updateOrdinalMeta();
    if (names) {
      for (var idx = start; idx < end; idx++) {
        var sourceIdx = idx - start;
        this._nameList[idx] = names[sourceIdx];
        if (shouldMakeIdFromName) {
          makeIdFromName(this, idx);
        }
      }
    }
  };
  SeriesData.prototype._updateOrdinalMeta = function () {
    var store = this._store;
    var dimensions = this.dimensions;
    for (var i = 0; i < dimensions.length; i++) {
      var dimInfo = this._dimInfos[dimensions[i]];
      if (dimInfo.ordinalMeta) {
        store.collectOrdinalMeta(dimInfo.storeDimIndex, dimInfo.ordinalMeta);
      }
    }
  };
  SeriesData.prototype._shouldMakeIdFromName = function () {
    var provider = this._store.getProvider();
    return this._idDimIdx == null && provider.getSource().sourceFormat !== SOURCE_FORMAT_TYPED_ARRAY && !provider.fillStorage;
  };
  SeriesData.prototype._doInit = function (start, end) {
    if (start >= end) {
      return;
    }
    var store = this._store;
    var provider = store.getProvider();
    this._updateOrdinalMeta();
    var nameList = this._nameList;
    var idList = this._idList;
    var sourceFormat = provider.getSource().sourceFormat;
    var isFormatOriginal = sourceFormat === SOURCE_FORMAT_ORIGINAL;
    // Each data item is value
    // [1, 2]
    // 2
    // Bar chart, line chart which uses category axis
    // only gives the 'y' value. 'x' value is the indices of category
    // Use a tempValue to normalize the value to be a (x, y) value
    // If dataItem is {name: ...} or {id: ...}, it has highest priority.
    // This kind of ids and names are always stored `_nameList` and `_idList`.
    if (isFormatOriginal && !provider.pure) {
      var sharedDataItem = [];
      for (var idx = start; idx < end; idx++) {
        // NOTICE: Try not to write things into dataItem
        var dataItem = provider.getItem(idx, sharedDataItem);
        if (!this.hasItemOption && isDataItemOption(dataItem)) {
          this.hasItemOption = true;
        }
        if (dataItem) {
          var itemName = dataItem.name;
          if (nameList[idx] == null && itemName != null) {
            nameList[idx] = convertOptionIdName(itemName, null);
          }
          var itemId = dataItem.id;
          if (idList[idx] == null && itemId != null) {
            idList[idx] = convertOptionIdName(itemId, null);
          }
        }
      }
    }
    if (this._shouldMakeIdFromName()) {
      for (var idx = start; idx < end; idx++) {
        makeIdFromName(this, idx);
      }
    }
    prepareInvertedIndex(this);
  };
  /**
   * PENDING: In fact currently this function is only used to short-circuit
   * the calling of `scale.unionExtentFromData` when data have been filtered by modules
   * like "dataZoom". `scale.unionExtentFromData` is used to calculate data extent for series on
   * an axis, but if a "axis related data filter module" is used, the extent of the axis have
   * been fixed and no need to calling `scale.unionExtentFromData` actually.
   * But if we add "custom data filter" in future, which is not "axis related", this method may
   * be still needed.
   *
   * Optimize for the scenario that data is filtered by a given extent.
   * Consider that if data amount is more than hundreds of thousand,
   * extent calculation will cost more than 10ms and the cache will
   * be erased because of the filtering.
   */
  SeriesData.prototype.getApproximateExtent = function (dim) {
    return this._approximateExtent[dim] || this._store.getDataExtent(this._getStoreDimIndex(dim));
  };
  /**
   * Calculate extent on a filtered data might be time consuming.
   * Approximate extent is only used for: calculate extent of filtered data outside.
   */
  SeriesData.prototype.setApproximateExtent = function (extent, dim) {
    dim = this.getDimension(dim);
    this._approximateExtent[dim] = extent.slice();
  };
  SeriesData.prototype.getCalculationInfo = function (key) {
    return this._calculationInfo[key];
  };
  SeriesData.prototype.setCalculationInfo = function (key, value) {
    isObject(key) ? zrUtil.extend(this._calculationInfo, key) : this._calculationInfo[key] = value;
  };
  /**
   * @return Never be null/undefined. `number` will be converted to string. Because:
   * In most cases, name is used in display, where returning a string is more convenient.
   * In other cases, name is used in query (see `indexOfName`), where we can keep the
   * rule that name `2` equals to name `'2'`.
   */
  SeriesData.prototype.getName = function (idx) {
    var rawIndex = this.getRawIndex(idx);
    var name = this._nameList[rawIndex];
    if (name == null && this._nameDimIdx != null) {
      name = getIdNameFromStore(this, this._nameDimIdx, rawIndex);
    }
    if (name == null) {
      name = '';
    }
    return name;
  };
  SeriesData.prototype._getCategory = function (dimIdx, idx) {
    var ordinal = this._store.get(dimIdx, idx);
    var ordinalMeta = this._store.getOrdinalMeta(dimIdx);
    if (ordinalMeta) {
      return ordinalMeta.categories[ordinal];
    }
    return ordinal;
  };
  /**
   * @return Never null/undefined. `number` will be converted to string. Because:
   * In all cases having encountered at present, id is used in making diff comparison, which
   * are usually based on hash map. We can keep the rule that the internal id are always string
   * (treat `2` is the same as `'2'`) to make the related logic simple.
   */
  SeriesData.prototype.getId = function (idx) {
    return getId(this, this.getRawIndex(idx));
  };
  SeriesData.prototype.count = function () {
    return this._store.count();
  };
  /**
   * Get value. Return NaN if idx is out of range.
   *
   * @notice Should better to use `data.getStore().get(dimIndex, dataIdx)` instead.
   */
  SeriesData.prototype.get = function (dim, idx) {
    var store = this._store;
    var dimInfo = this._dimInfos[dim];
    if (dimInfo) {
      return store.get(dimInfo.storeDimIndex, idx);
    }
  };
  /**
   * @notice Should better to use `data.getStore().getByRawIndex(dimIndex, dataIdx)` instead.
   */
  SeriesData.prototype.getByRawIndex = function (dim, rawIdx) {
    var store = this._store;
    var dimInfo = this._dimInfos[dim];
    if (dimInfo) {
      return store.getByRawIndex(dimInfo.storeDimIndex, rawIdx);
    }
  };
  SeriesData.prototype.getIndices = function () {
    return this._store.getIndices();
  };
  SeriesData.prototype.getDataExtent = function (dim) {
    return this._store.getDataExtent(this._getStoreDimIndex(dim));
  };
  SeriesData.prototype.getSum = function (dim) {
    return this._store.getSum(this._getStoreDimIndex(dim));
  };
  SeriesData.prototype.getMedian = function (dim) {
    return this._store.getMedian(this._getStoreDimIndex(dim));
  };
  SeriesData.prototype.getValues = function (dimensions, idx) {
    var _this = this;
    var store = this._store;
    return zrUtil.isArray(dimensions) ? store.getValues(map(dimensions, function (dim) {
      return _this._getStoreDimIndex(dim);
    }), idx) : store.getValues(dimensions);
  };
  /**
   * If value is NaN. Including '-'
   * Only check the coord dimensions.
   */
  SeriesData.prototype.hasValue = function (idx) {
    var dataDimIndicesOnCoord = this._dimSummary.dataDimIndicesOnCoord;
    for (var i = 0, len = dataDimIndicesOnCoord.length; i < len; i++) {
      // Ordinal type originally can be string or number.
      // But when an ordinal type is used on coord, it can
      // not be string but only number. So we can also use isNaN.
      if (isNaN(this._store.get(dataDimIndicesOnCoord[i], idx))) {
        return false;
      }
    }
    return true;
  };
  /**
   * Retrieve the index with given name
   */
  SeriesData.prototype.indexOfName = function (name) {
    for (var i = 0, len = this._store.count(); i < len; i++) {
      if (this.getName(i) === name) {
        return i;
      }
    }
    return -1;
  };
  SeriesData.prototype.getRawIndex = function (idx) {
    return this._store.getRawIndex(idx);
  };
  SeriesData.prototype.indexOfRawIndex = function (rawIndex) {
    return this._store.indexOfRawIndex(rawIndex);
  };
  /**
   * Only support the dimension which inverted index created.
   * Do not support other cases until required.
   * @param dim concrete dim
   * @param value ordinal index
   * @return rawIndex
   */
  SeriesData.prototype.rawIndexOf = function (dim, value) {
    var invertedIndices = dim && this._invertedIndicesMap[dim];
    if (process.env.NODE_ENV !== 'production') {
      if (!invertedIndices) {
        throw new Error('Do not supported yet');
      }
    }
    var rawIndex = invertedIndices && invertedIndices[value];
    if (rawIndex == null || isNaN(rawIndex)) {
      return INDEX_NOT_FOUND;
    }
    return rawIndex;
  };
  /**
   * Retrieve the index of nearest value
   * @param dim
   * @param value
   * @param [maxDistance=Infinity]
   * @return If and only if multiple indices has
   *         the same value, they are put to the result.
   */
  SeriesData.prototype.indicesOfNearest = function (dim, value, maxDistance) {
    return this._store.indicesOfNearest(this._getStoreDimIndex(dim), value, maxDistance);
  };
  SeriesData.prototype.each = function (dims, cb, ctx) {
    'use strict';

    if (zrUtil.isFunction(dims)) {
      ctx = cb;
      cb = dims;
      dims = [];
    }
    // ctxCompat just for compat echarts3
    var fCtx = ctx || this;
    var dimIndices = map(normalizeDimensions(dims), this._getStoreDimIndex, this);
    this._store.each(dimIndices, fCtx ? zrUtil.bind(cb, fCtx) : cb);
  };
  SeriesData.prototype.filterSelf = function (dims, cb, ctx) {
    'use strict';

    if (zrUtil.isFunction(dims)) {
      ctx = cb;
      cb = dims;
      dims = [];
    }
    // ctxCompat just for compat echarts3
    var fCtx = ctx || this;
    var dimIndices = map(normalizeDimensions(dims), this._getStoreDimIndex, this);
    this._store = this._store.filter(dimIndices, fCtx ? zrUtil.bind(cb, fCtx) : cb);
    return this;
  };
  /**
   * Select data in range. (For optimization of filter)
   * (Manually inline code, support 5 million data filtering in data zoom.)
   */
  SeriesData.prototype.selectRange = function (range) {
    'use strict';

    var _this = this;
    var innerRange = {};
    var dims = zrUtil.keys(range);
    var dimIndices = [];
    zrUtil.each(dims, function (dim) {
      var dimIdx = _this._getStoreDimIndex(dim);
      innerRange[dimIdx] = range[dim];
      dimIndices.push(dimIdx);
    });
    this._store = this._store.selectRange(innerRange);
    return this;
  };
  /* eslint-enable max-len */
  SeriesData.prototype.mapArray = function (dims, cb, ctx) {
    'use strict';

    if (zrUtil.isFunction(dims)) {
      ctx = cb;
      cb = dims;
      dims = [];
    }
    // ctxCompat just for compat echarts3
    ctx = ctx || this;
    var result = [];
    this.each(dims, function () {
      result.push(cb && cb.apply(this, arguments));
    }, ctx);
    return result;
  };
  SeriesData.prototype.map = function (dims, cb, ctx, ctxCompat) {
    'use strict';

    // ctxCompat just for compat echarts3
    var fCtx = ctx || ctxCompat || this;
    var dimIndices = map(normalizeDimensions(dims), this._getStoreDimIndex, this);
    var list = cloneListForMapAndSample(this);
    list._store = this._store.map(dimIndices, fCtx ? zrUtil.bind(cb, fCtx) : cb);
    return list;
  };
  SeriesData.prototype.modify = function (dims, cb, ctx, ctxCompat) {
    var _this = this;
    // ctxCompat just for compat echarts3
    var fCtx = ctx || ctxCompat || this;
    if (process.env.NODE_ENV !== 'production') {
      zrUtil.each(normalizeDimensions(dims), function (dim) {
        var dimInfo = _this.getDimensionInfo(dim);
        if (!dimInfo.isCalculationCoord) {
          console.error('Danger: only stack dimension can be modified');
        }
      });
    }
    var dimIndices = map(normalizeDimensions(dims), this._getStoreDimIndex, this);
    // If do shallow clone here, if there are too many stacked series,
    // it still cost lots of memory, because `_store.dimensions` are not shared.
    // We should consider there probably be shallow clone happen in each series
    // in consequent filter/map.
    this._store.modify(dimIndices, fCtx ? zrUtil.bind(cb, fCtx) : cb);
  };
  /**
   * Large data down sampling on given dimension
   * @param sampleIndex Sample index for name and id
   */
  SeriesData.prototype.downSample = function (dimension, rate, sampleValue, sampleIndex) {
    var list = cloneListForMapAndSample(this);
    list._store = this._store.downSample(this._getStoreDimIndex(dimension), rate, sampleValue, sampleIndex);
    return list;
  };
  /**
   * Large data down sampling using min-max
   * @param {string} valueDimension
   * @param {number} rate
   */
  SeriesData.prototype.minmaxDownSample = function (valueDimension, rate) {
    var list = cloneListForMapAndSample(this);
    list._store = this._store.minmaxDownSample(this._getStoreDimIndex(valueDimension), rate);
    return list;
  };
  /**
   * Large data down sampling using largest-triangle-three-buckets
   * @param {string} valueDimension
   * @param {number} targetCount
   */
  SeriesData.prototype.lttbDownSample = function (valueDimension, rate) {
    var list = cloneListForMapAndSample(this);
    list._store = this._store.lttbDownSample(this._getStoreDimIndex(valueDimension), rate);
    return list;
  };
  SeriesData.prototype.getRawDataItem = function (idx) {
    return this._store.getRawDataItem(idx);
  };
  /**
   * Get model of one data item.
   */
  // TODO: Type of data item
  SeriesData.prototype.getItemModel = function (idx) {
    var hostModel = this.hostModel;
    var dataItem = this.getRawDataItem(idx);
    return new Model(dataItem, hostModel, hostModel && hostModel.ecModel);
  };
  /**
   * Create a data differ
   */
  SeriesData.prototype.diff = function (otherList) {
    var thisList = this;
    return new DataDiffer(otherList ? otherList.getStore().getIndices() : [], this.getStore().getIndices(), function (idx) {
      return getId(otherList, idx);
    }, function (idx) {
      return getId(thisList, idx);
    });
  };
  /**
   * Get visual property.
   */
  SeriesData.prototype.getVisual = function (key) {
    var visual = this._visual;
    return visual && visual[key];
  };
  SeriesData.prototype.setVisual = function (kvObj, val) {
    this._visual = this._visual || {};
    if (isObject(kvObj)) {
      zrUtil.extend(this._visual, kvObj);
    } else {
      this._visual[kvObj] = val;
    }
  };
  /**
   * Get visual property of single data item
   */
  // eslint-disable-next-line
  SeriesData.prototype.getItemVisual = function (idx, key) {
    var itemVisual = this._itemVisuals[idx];
    var val = itemVisual && itemVisual[key];
    if (val == null) {
      // Use global visual property
      return this.getVisual(key);
    }
    return val;
  };
  /**
   * If exists visual property of single data item
   */
  SeriesData.prototype.hasItemVisual = function () {
    return this._itemVisuals.length > 0;
  };
  /**
   * Make sure itemVisual property is unique
   */
  // TODO: use key to save visual to reduce memory.
  SeriesData.prototype.ensureUniqueItemVisual = function (idx, key) {
    var itemVisuals = this._itemVisuals;
    var itemVisual = itemVisuals[idx];
    if (!itemVisual) {
      itemVisual = itemVisuals[idx] = {};
    }
    var val = itemVisual[key];
    if (val == null) {
      val = this.getVisual(key);
      // TODO Performance?
      if (zrUtil.isArray(val)) {
        val = val.slice();
      } else if (isObject(val)) {
        val = zrUtil.extend({}, val);
      }
      itemVisual[key] = val;
    }
    return val;
  };
  // eslint-disable-next-line
  SeriesData.prototype.setItemVisual = function (idx, key, value) {
    var itemVisual = this._itemVisuals[idx] || {};
    this._itemVisuals[idx] = itemVisual;
    if (isObject(key)) {
      zrUtil.extend(itemVisual, key);
    } else {
      itemVisual[key] = value;
    }
  };
  /**
   * Clear itemVisuals and list visual.
   */
  SeriesData.prototype.clearAllVisual = function () {
    this._visual = {};
    this._itemVisuals = [];
  };
  SeriesData.prototype.setLayout = function (key, val) {
    isObject(key) ? zrUtil.extend(this._layout, key) : this._layout[key] = val;
  };
  /**
   * Get layout property.
   */
  SeriesData.prototype.getLayout = function (key) {
    return this._layout[key];
  };
  /**
   * Get layout of single data item
   */
  SeriesData.prototype.getItemLayout = function (idx) {
    return this._itemLayouts[idx];
  };
  /**
   * Set layout of single data item
   */
  SeriesData.prototype.setItemLayout = function (idx, layout, merge) {
    this._itemLayouts[idx] = merge ? zrUtil.extend(this._itemLayouts[idx] || {}, layout) : layout;
  };
  /**
   * Clear all layout of single data item
   */
  SeriesData.prototype.clearItemLayouts = function () {
    this._itemLayouts.length = 0;
  };
  /**
   * Set graphic element relative to data. It can be set as null
   */
  SeriesData.prototype.setItemGraphicEl = function (idx, el) {
    var seriesIndex = this.hostModel && this.hostModel.seriesIndex;
    setCommonECData(seriesIndex, this.dataType, idx, el);
    this._graphicEls[idx] = el;
  };
  SeriesData.prototype.getItemGraphicEl = function (idx) {
    return this._graphicEls[idx];
  };
  SeriesData.prototype.eachItemGraphicEl = function (cb, context) {
    zrUtil.each(this._graphicEls, function (el, idx) {
      if (el) {
        cb && cb.call(context, el, idx);
      }
    });
  };
  /**
   * Shallow clone a new list except visual and layout properties, and graph elements.
   * New list only change the indices.
   */
  SeriesData.prototype.cloneShallow = function (list) {
    if (!list) {
      list = new SeriesData(this._schema ? this._schema : map(this.dimensions, this._getDimInfo, this), this.hostModel);
    }
    transferProperties(list, this);
    list._store = this._store;
    return list;
  };
  /**
   * Wrap some method to add more feature
   */
  SeriesData.prototype.wrapMethod = function (methodName, injectFunction) {
    var originalMethod = this[methodName];
    if (!zrUtil.isFunction(originalMethod)) {
      return;
    }
    this.__wrappedMethods = this.__wrappedMethods || [];
    this.__wrappedMethods.push(methodName);
    this[methodName] = function () {
      var res = originalMethod.apply(this, arguments);
      return injectFunction.apply(this, [res].concat(zrUtil.slice(arguments)));
    };
  };
  // ----------------------------------------------------------
  // A work around for internal method visiting private member.
  // ----------------------------------------------------------
  SeriesData.internalField = function () {
    prepareInvertedIndex = function (data) {
      var invertedIndicesMap = data._invertedIndicesMap;
      zrUtil.each(invertedIndicesMap, function (invertedIndices, dim) {
        var dimInfo = data._dimInfos[dim];
        // Currently, only dimensions that has ordinalMeta can create inverted indices.
        var ordinalMeta = dimInfo.ordinalMeta;
        var store = data._store;
        if (ordinalMeta) {
          invertedIndices = invertedIndicesMap[dim] = new CtorInt32Array(ordinalMeta.categories.length);
          // The default value of TypedArray is 0. To avoid miss
          // mapping to 0, we should set it as INDEX_NOT_FOUND.
          for (var i = 0; i < invertedIndices.length; i++) {
            invertedIndices[i] = INDEX_NOT_FOUND;
          }
          for (var i = 0; i < store.count(); i++) {
            // Only support the case that all values are distinct.
            invertedIndices[store.get(dimInfo.storeDimIndex, i)] = i;
          }
        }
      });
    };
    getIdNameFromStore = function (data, dimIdx, idx) {
      return convertOptionIdName(data._getCategory(dimIdx, idx), null);
    };
    /**
     * @see the comment of `List['getId']`.
     */
    getId = function (data, rawIndex) {
      var id = data._idList[rawIndex];
      if (id == null && data._idDimIdx != null) {
        id = getIdNameFromStore(data, data._idDimIdx, rawIndex);
      }
      if (id == null) {
        id = ID_PREFIX + rawIndex;
      }
      return id;
    };
    normalizeDimensions = function (dimensions) {
      if (!zrUtil.isArray(dimensions)) {
        dimensions = dimensions != null ? [dimensions] : [];
      }
      return dimensions;
    };
    /**
     * Data in excludeDimensions is copied, otherwise transferred.
     */
    cloneListForMapAndSample = function (original) {
      var list = new SeriesData(original._schema ? original._schema : map(original.dimensions, original._getDimInfo, original), original.hostModel);
      // FIXME If needs stackedOn, value may already been stacked
      transferProperties(list, original);
      return list;
    };
    transferProperties = function (target, source) {
      zrUtil.each(TRANSFERABLE_PROPERTIES.concat(source.__wrappedMethods || []), function (propName) {
        if (source.hasOwnProperty(propName)) {
          target[propName] = source[propName];
        }
      });
      target.__wrappedMethods = source.__wrappedMethods;
      zrUtil.each(CLONE_PROPERTIES, function (propName) {
        target[propName] = zrUtil.clone(source[propName]);
      });
      target._calculationInfo = zrUtil.extend({}, source._calculationInfo);
    };
    makeIdFromName = function (data, idx) {
      var nameList = data._nameList;
      var idList = data._idList;
      var nameDimIdx = data._nameDimIdx;
      var idDimIdx = data._idDimIdx;
      var name = nameList[idx];
      var id = idList[idx];
      if (name == null && nameDimIdx != null) {
        nameList[idx] = name = getIdNameFromStore(data, nameDimIdx, idx);
      }
      if (id == null && idDimIdx != null) {
        idList[idx] = id = getIdNameFromStore(data, idDimIdx, idx);
      }
      if (id == null && name != null) {
        var nameRepeatCount = data._nameRepeatCount;
        var nmCnt = nameRepeatCount[name] = (nameRepeatCount[name] || 0) + 1;
        id = name;
        if (nmCnt > 1) {
          id += '__ec__' + nmCnt;
        }
        idList[idx] = id;
      }
    };
  }();
  return SeriesData;
}();
export default SeriesData;