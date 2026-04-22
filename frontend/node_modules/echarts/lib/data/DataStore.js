
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
import { assert, clone, createHashMap, isFunction, keys, map, reduce } from 'zrender/lib/core/util.js';
import { parseDataValue } from './helper/dataValueHelper.js';
import { shouldRetrieveDataByName } from './Source.js';
var UNDEFINED = 'undefined';
/* global Float64Array, Int32Array, Uint32Array, Uint16Array */
// Caution: MUST not use `new CtorUint32Array(arr, 0, len)`, because the Ctor of array is
// different from the Ctor of typed array.
export var CtorUint32Array = typeof Uint32Array === UNDEFINED ? Array : Uint32Array;
export var CtorUint16Array = typeof Uint16Array === UNDEFINED ? Array : Uint16Array;
export var CtorInt32Array = typeof Int32Array === UNDEFINED ? Array : Int32Array;
export var CtorFloat64Array = typeof Float64Array === UNDEFINED ? Array : Float64Array;
/**
 * Multi dimensional data store
 */
var dataCtors = {
  'float': CtorFloat64Array,
  'int': CtorInt32Array,
  // Ordinal data type can be string or int
  'ordinal': Array,
  'number': Array,
  'time': CtorFloat64Array
};
var defaultDimValueGetters;
function getIndicesCtor(rawCount) {
  // The possible max value in this._indicies is always this._rawCount despite of filtering.
  return rawCount > 65535 ? CtorUint32Array : CtorUint16Array;
}
;
function getInitialExtent() {
  return [Infinity, -Infinity];
}
;
function cloneChunk(originalChunk) {
  var Ctor = originalChunk.constructor;
  // Only shallow clone is enough when Array.
  return Ctor === Array ? originalChunk.slice() : new Ctor(originalChunk);
}
function prepareStore(store, dimIdx, dimType, end, append) {
  var DataCtor = dataCtors[dimType || 'float'];
  if (append) {
    var oldStore = store[dimIdx];
    var oldLen = oldStore && oldStore.length;
    if (!(oldLen === end)) {
      var newStore = new DataCtor(end);
      // The cost of the copy is probably inconsiderable
      // within the initial chunkSize.
      for (var j = 0; j < oldLen; j++) {
        newStore[j] = oldStore[j];
      }
      store[dimIdx] = newStore;
    }
  } else {
    store[dimIdx] = new DataCtor(end);
  }
}
;
/**
 * Basically, DataStore API keep immutable.
 */
var DataStore = /** @class */function () {
  function DataStore() {
    this._chunks = [];
    // It will not be calculated until needed.
    this._rawExtent = [];
    this._extent = [];
    this._count = 0;
    this._rawCount = 0;
    this._calcDimNameToIdx = createHashMap();
  }
  /**
   * Initialize from data
   */
  DataStore.prototype.initData = function (provider, inputDimensions, dimValueGetter) {
    if (process.env.NODE_ENV !== 'production') {
      assert(isFunction(provider.getItem) && isFunction(provider.count), 'Invalid data provider.');
    }
    this._provider = provider;
    // Clear
    this._chunks = [];
    this._indices = null;
    this.getRawIndex = this._getRawIdxIdentity;
    var source = provider.getSource();
    var defaultGetter = this.defaultDimValueGetter = defaultDimValueGetters[source.sourceFormat];
    // Default dim value getter
    this._dimValueGetter = dimValueGetter || defaultGetter;
    // Reset raw extent.
    this._rawExtent = [];
    var willRetrieveDataByName = shouldRetrieveDataByName(source);
    this._dimensions = map(inputDimensions, function (dim) {
      if (process.env.NODE_ENV !== 'production') {
        if (willRetrieveDataByName) {
          assert(dim.property != null);
        }
      }
      return {
        // Only pick these two props. Not leak other properties like orderMeta.
        type: dim.type,
        property: dim.property
      };
    });
    this._initDataFromProvider(0, provider.count());
  };
  DataStore.prototype.getProvider = function () {
    return this._provider;
  };
  /**
   * Caution: even when a `source` instance owned by a series, the created data store
   * may still be shared by different sereis (the source hash does not use all `source`
   * props, see `sourceManager`). In this case, the `source` props that are not used in
   * hash (like `source.dimensionDefine`) probably only belongs to a certain series and
   * thus should not be fetch here.
   */
  DataStore.prototype.getSource = function () {
    return this._provider.getSource();
  };
  /**
   * @caution Only used in dataStack.
   */
  DataStore.prototype.ensureCalculationDimension = function (dimName, type) {
    var calcDimNameToIdx = this._calcDimNameToIdx;
    var dimensions = this._dimensions;
    var calcDimIdx = calcDimNameToIdx.get(dimName);
    if (calcDimIdx != null) {
      if (dimensions[calcDimIdx].type === type) {
        return calcDimIdx;
      }
    } else {
      calcDimIdx = dimensions.length;
    }
    dimensions[calcDimIdx] = {
      type: type
    };
    calcDimNameToIdx.set(dimName, calcDimIdx);
    this._chunks[calcDimIdx] = new dataCtors[type || 'float'](this._rawCount);
    this._rawExtent[calcDimIdx] = getInitialExtent();
    return calcDimIdx;
  };
  DataStore.prototype.collectOrdinalMeta = function (dimIdx, ordinalMeta) {
    var chunk = this._chunks[dimIdx];
    var dim = this._dimensions[dimIdx];
    var rawExtents = this._rawExtent;
    var offset = dim.ordinalOffset || 0;
    var len = chunk.length;
    if (offset === 0) {
      // We need to reset the rawExtent if collect is from start.
      // Because this dimension may be guessed as number and calcuating a wrong extent.
      rawExtents[dimIdx] = getInitialExtent();
    }
    var dimRawExtent = rawExtents[dimIdx];
    // Parse from previous data offset. len may be changed after appendData
    for (var i = offset; i < len; i++) {
      var val = chunk[i] = ordinalMeta.parseAndCollect(chunk[i]);
      if (!isNaN(val)) {
        dimRawExtent[0] = Math.min(val, dimRawExtent[0]);
        dimRawExtent[1] = Math.max(val, dimRawExtent[1]);
      }
    }
    dim.ordinalMeta = ordinalMeta;
    dim.ordinalOffset = len;
    dim.type = 'ordinal'; // Force to be ordinal
  };
  DataStore.prototype.getOrdinalMeta = function (dimIdx) {
    var dimInfo = this._dimensions[dimIdx];
    var ordinalMeta = dimInfo.ordinalMeta;
    return ordinalMeta;
  };
  DataStore.prototype.getDimensionProperty = function (dimIndex) {
    var item = this._dimensions[dimIndex];
    return item && item.property;
  };
  /**
   * Caution: Can be only called on raw data (before `this._indices` created).
   */
  DataStore.prototype.appendData = function (data) {
    if (process.env.NODE_ENV !== 'production') {
      assert(!this._indices, 'appendData can only be called on raw data.');
    }
    var provider = this._provider;
    var start = this.count();
    provider.appendData(data);
    var end = provider.count();
    if (!provider.persistent) {
      end += start;
    }
    if (start < end) {
      this._initDataFromProvider(start, end, true);
    }
    return [start, end];
  };
  DataStore.prototype.appendValues = function (values, minFillLen) {
    var chunks = this._chunks;
    var dimensions = this._dimensions;
    var dimLen = dimensions.length;
    var rawExtent = this._rawExtent;
    var start = this.count();
    var end = start + Math.max(values.length, minFillLen || 0);
    for (var i = 0; i < dimLen; i++) {
      var dim = dimensions[i];
      prepareStore(chunks, i, dim.type, end, true);
    }
    var emptyDataItem = [];
    for (var idx = start; idx < end; idx++) {
      var sourceIdx = idx - start;
      // Store the data by dimensions
      for (var dimIdx = 0; dimIdx < dimLen; dimIdx++) {
        var dim = dimensions[dimIdx];
        var val = defaultDimValueGetters.arrayRows.call(this, values[sourceIdx] || emptyDataItem, dim.property, sourceIdx, dimIdx);
        chunks[dimIdx][idx] = val;
        var dimRawExtent = rawExtent[dimIdx];
        val < dimRawExtent[0] && (dimRawExtent[0] = val);
        val > dimRawExtent[1] && (dimRawExtent[1] = val);
      }
    }
    this._rawCount = this._count = end;
    return {
      start: start,
      end: end
    };
  };
  DataStore.prototype._initDataFromProvider = function (start, end, append) {
    var provider = this._provider;
    var chunks = this._chunks;
    var dimensions = this._dimensions;
    var dimLen = dimensions.length;
    var rawExtent = this._rawExtent;
    var dimNames = map(dimensions, function (dim) {
      return dim.property;
    });
    for (var i = 0; i < dimLen; i++) {
      var dim = dimensions[i];
      if (!rawExtent[i]) {
        rawExtent[i] = getInitialExtent();
      }
      prepareStore(chunks, i, dim.type, end, append);
    }
    if (provider.fillStorage) {
      provider.fillStorage(start, end, chunks, rawExtent);
    } else {
      var dataItem = [];
      for (var idx = start; idx < end; idx++) {
        // NOTICE: Try not to write things into dataItem
        dataItem = provider.getItem(idx, dataItem);
        // Each data item is value
        // [1, 2]
        // 2
        // Bar chart, line chart which uses category axis
        // only gives the 'y' value. 'x' value is the indices of category
        // Use a tempValue to normalize the value to be a (x, y) value
        // Store the data by dimensions
        for (var dimIdx = 0; dimIdx < dimLen; dimIdx++) {
          var dimStorage = chunks[dimIdx];
          // PENDING NULL is empty or zero
          var val = this._dimValueGetter(dataItem, dimNames[dimIdx], idx, dimIdx);
          dimStorage[idx] = val;
          var dimRawExtent = rawExtent[dimIdx];
          val < dimRawExtent[0] && (dimRawExtent[0] = val);
          val > dimRawExtent[1] && (dimRawExtent[1] = val);
        }
      }
    }
    if (!provider.persistent && provider.clean) {
      // Clean unused data if data source is typed array.
      provider.clean();
    }
    this._rawCount = this._count = end;
    // Reset data extent
    this._extent = [];
  };
  DataStore.prototype.count = function () {
    return this._count;
  };
  /**
   * Get value. Return NaN if idx is out of range.
   */
  DataStore.prototype.get = function (dim, idx) {
    if (!(idx >= 0 && idx < this._count)) {
      return NaN;
    }
    var dimStore = this._chunks[dim];
    return dimStore ? dimStore[this.getRawIndex(idx)] : NaN;
  };
  DataStore.prototype.getValues = function (dimensions, idx) {
    var values = [];
    var dimArr = [];
    if (idx == null) {
      idx = dimensions;
      // TODO get all from store?
      dimensions = [];
      // All dimensions
      for (var i = 0; i < this._dimensions.length; i++) {
        dimArr.push(i);
      }
    } else {
      dimArr = dimensions;
    }
    for (var i = 0, len = dimArr.length; i < len; i++) {
      values.push(this.get(dimArr[i], idx));
    }
    return values;
  };
  /**
   * @param dim concrete dim
   */
  DataStore.prototype.getByRawIndex = function (dim, rawIdx) {
    if (!(rawIdx >= 0 && rawIdx < this._rawCount)) {
      return NaN;
    }
    var dimStore = this._chunks[dim];
    return dimStore ? dimStore[rawIdx] : NaN;
  };
  /**
   * Get sum of data in one dimension
   */
  DataStore.prototype.getSum = function (dim) {
    var dimData = this._chunks[dim];
    var sum = 0;
    if (dimData) {
      for (var i = 0, len = this.count(); i < len; i++) {
        var value = this.get(dim, i);
        if (!isNaN(value)) {
          sum += value;
        }
      }
    }
    return sum;
  };
  /**
   * Get median of data in one dimension
   */
  DataStore.prototype.getMedian = function (dim) {
    var dimDataArray = [];
    // map all data of one dimension
    this.each([dim], function (val) {
      if (!isNaN(val)) {
        dimDataArray.push(val);
      }
    });
    // TODO
    // Use quick select?
    var sortedDimDataArray = dimDataArray.sort(function (a, b) {
      return a - b;
    });
    var len = this.count();
    // calculate median
    return len === 0 ? 0 : len % 2 === 1 ? sortedDimDataArray[(len - 1) / 2] : (sortedDimDataArray[len / 2] + sortedDimDataArray[len / 2 - 1]) / 2;
  };
  /**
   * Retrieve the index with given raw data index.
   */
  DataStore.prototype.indexOfRawIndex = function (rawIndex) {
    if (rawIndex >= this._rawCount || rawIndex < 0) {
      return -1;
    }
    if (!this._indices) {
      return rawIndex;
    }
    // Indices are ascending
    var indices = this._indices;
    // If rawIndex === dataIndex
    var rawDataIndex = indices[rawIndex];
    if (rawDataIndex != null && rawDataIndex < this._count && rawDataIndex === rawIndex) {
      return rawIndex;
    }
    var left = 0;
    var right = this._count - 1;
    while (left <= right) {
      var mid = (left + right) / 2 | 0;
      if (indices[mid] < rawIndex) {
        left = mid + 1;
      } else if (indices[mid] > rawIndex) {
        right = mid - 1;
      } else {
        return mid;
      }
    }
    return -1;
  };
  /**
   * Retrieve the index of nearest value.
   * @param dim
   * @param value
   * @param [maxDistance=Infinity]
   * @return If and only if multiple indices have
   *         the same value, they are put to the result.
   */
  DataStore.prototype.indicesOfNearest = function (dim, value, maxDistance) {
    var chunks = this._chunks;
    var dimData = chunks[dim];
    var nearestIndices = [];
    if (!dimData) {
      return nearestIndices;
    }
    if (maxDistance == null) {
      maxDistance = Infinity;
    }
    var minDist = Infinity;
    var minDiff = -1;
    var nearestIndicesLen = 0;
    // Check the test case of `test/ut/spec/data/SeriesData.js`.
    for (var i = 0, len = this.count(); i < len; i++) {
      var dataIndex = this.getRawIndex(i);
      var diff = value - dimData[dataIndex];
      var dist = Math.abs(diff);
      if (dist <= maxDistance) {
        // When the `value` is at the middle of `this.get(dim, i)` and `this.get(dim, i+1)`,
        // we'd better not push both of them to `nearestIndices`, otherwise it is easy to
        // get more than one item in `nearestIndices` (more specifically, in `tooltip`).
        // So we choose the one that `diff >= 0` in this case.
        // But if `this.get(dim, i)` and `this.get(dim, j)` get the same value, both of them
        // should be push to `nearestIndices`.
        if (dist < minDist || dist === minDist && diff >= 0 && minDiff < 0) {
          minDist = dist;
          minDiff = diff;
          nearestIndicesLen = 0;
        }
        if (diff === minDiff) {
          nearestIndices[nearestIndicesLen++] = i;
        }
      }
    }
    nearestIndices.length = nearestIndicesLen;
    return nearestIndices;
  };
  DataStore.prototype.getIndices = function () {
    var newIndices;
    var indices = this._indices;
    if (indices) {
      var Ctor = indices.constructor;
      var thisCount = this._count;
      // `new Array(a, b, c)` is different from `new Uint32Array(a, b, c)`.
      if (Ctor === Array) {
        newIndices = new Ctor(thisCount);
        for (var i = 0; i < thisCount; i++) {
          newIndices[i] = indices[i];
        }
      } else {
        newIndices = new Ctor(indices.buffer, 0, thisCount);
      }
    } else {
      var Ctor = getIndicesCtor(this._rawCount);
      newIndices = new Ctor(this.count());
      for (var i = 0; i < newIndices.length; i++) {
        newIndices[i] = i;
      }
    }
    return newIndices;
  };
  /**
   * Data filter.
   */
  DataStore.prototype.filter = function (dims, cb) {
    if (!this._count) {
      return this;
    }
    var newStore = this.clone();
    var count = newStore.count();
    var Ctor = getIndicesCtor(newStore._rawCount);
    var newIndices = new Ctor(count);
    var value = [];
    var dimSize = dims.length;
    var offset = 0;
    var dim0 = dims[0];
    var chunks = newStore._chunks;
    for (var i = 0; i < count; i++) {
      var keep = void 0;
      var rawIdx = newStore.getRawIndex(i);
      // Simple optimization
      if (dimSize === 0) {
        keep = cb(i);
      } else if (dimSize === 1) {
        var val = chunks[dim0][rawIdx];
        keep = cb(val, i);
      } else {
        var k = 0;
        for (; k < dimSize; k++) {
          value[k] = chunks[dims[k]][rawIdx];
        }
        value[k] = i;
        keep = cb.apply(null, value);
      }
      if (keep) {
        newIndices[offset++] = rawIdx;
      }
    }
    // Set indices after filtered.
    if (offset < count) {
      newStore._indices = newIndices;
    }
    newStore._count = offset;
    // Reset data extent
    newStore._extent = [];
    newStore._updateGetRawIdx();
    return newStore;
  };
  /**
   * Select data in range. (For optimization of filter)
   * (Manually inline code, support 5 million data filtering in data zoom.)
   */
  DataStore.prototype.selectRange = function (range) {
    var newStore = this.clone();
    var len = newStore._count;
    if (!len) {
      return this;
    }
    var dims = keys(range);
    var dimSize = dims.length;
    if (!dimSize) {
      return this;
    }
    var originalCount = newStore.count();
    var Ctor = getIndicesCtor(newStore._rawCount);
    var newIndices = new Ctor(originalCount);
    var offset = 0;
    var dim0 = dims[0];
    var min = range[dim0][0];
    var max = range[dim0][1];
    var storeArr = newStore._chunks;
    var quickFinished = false;
    if (!newStore._indices) {
      // Extreme optimization for common case. About 2x faster in chrome.
      var idx = 0;
      if (dimSize === 1) {
        var dimStorage = storeArr[dims[0]];
        for (var i = 0; i < len; i++) {
          var val = dimStorage[i];
          // NaN will not be filtered. Consider the case, in line chart, empty
          // value indicates the line should be broken. But for the case like
          // scatter plot, a data item with empty value will not be rendered,
          // but the axis extent may be effected if some other dim of the data
          // item has value. Fortunately it is not a significant negative effect.
          if (val >= min && val <= max || isNaN(val)) {
            newIndices[offset++] = idx;
          }
          idx++;
        }
        quickFinished = true;
      } else if (dimSize === 2) {
        var dimStorage = storeArr[dims[0]];
        var dimStorage2 = storeArr[dims[1]];
        var min2 = range[dims[1]][0];
        var max2 = range[dims[1]][1];
        for (var i = 0; i < len; i++) {
          var val = dimStorage[i];
          var val2 = dimStorage2[i];
          // Do not filter NaN, see comment above.
          if ((val >= min && val <= max || isNaN(val)) && (val2 >= min2 && val2 <= max2 || isNaN(val2))) {
            newIndices[offset++] = idx;
          }
          idx++;
        }
        quickFinished = true;
      }
    }
    if (!quickFinished) {
      if (dimSize === 1) {
        for (var i = 0; i < originalCount; i++) {
          var rawIndex = newStore.getRawIndex(i);
          var val = storeArr[dims[0]][rawIndex];
          // Do not filter NaN, see comment above.
          if (val >= min && val <= max || isNaN(val)) {
            newIndices[offset++] = rawIndex;
          }
        }
      } else {
        for (var i = 0; i < originalCount; i++) {
          var keep = true;
          var rawIndex = newStore.getRawIndex(i);
          for (var k = 0; k < dimSize; k++) {
            var dimk = dims[k];
            var val = storeArr[dimk][rawIndex];
            // Do not filter NaN, see comment above.
            if (val < range[dimk][0] || val > range[dimk][1]) {
              keep = false;
            }
          }
          if (keep) {
            newIndices[offset++] = newStore.getRawIndex(i);
          }
        }
      }
    }
    // Set indices after filtered.
    if (offset < originalCount) {
      newStore._indices = newIndices;
    }
    newStore._count = offset;
    // Reset data extent
    newStore._extent = [];
    newStore._updateGetRawIdx();
    return newStore;
  };
  // /**
  //  * Data mapping to a plain array
  //  */
  // mapArray(dims: DimensionIndex[], cb: MapArrayCb): any[] {
  //     const result: any[] = [];
  //     this.each(dims, function () {
  //         result.push(cb && (cb as MapArrayCb).apply(null, arguments));
  //     });
  //     return result;
  // }
  /**
   * Data mapping to a new List with given dimensions
   */
  DataStore.prototype.map = function (dims, cb) {
    // TODO only clone picked chunks.
    var target = this.clone(dims);
    this._updateDims(target, dims, cb);
    return target;
  };
  /**
   * @caution Danger!! Only used in dataStack.
   */
  DataStore.prototype.modify = function (dims, cb) {
    this._updateDims(this, dims, cb);
  };
  DataStore.prototype._updateDims = function (target, dims, cb) {
    var targetChunks = target._chunks;
    var tmpRetValue = [];
    var dimSize = dims.length;
    var dataCount = target.count();
    var values = [];
    var rawExtent = target._rawExtent;
    for (var i = 0; i < dims.length; i++) {
      rawExtent[dims[i]] = getInitialExtent();
    }
    for (var dataIndex = 0; dataIndex < dataCount; dataIndex++) {
      var rawIndex = target.getRawIndex(dataIndex);
      for (var k = 0; k < dimSize; k++) {
        values[k] = targetChunks[dims[k]][rawIndex];
      }
      values[dimSize] = dataIndex;
      var retValue = cb && cb.apply(null, values);
      if (retValue != null) {
        // a number or string (in oridinal dimension)?
        if (typeof retValue !== 'object') {
          tmpRetValue[0] = retValue;
          retValue = tmpRetValue;
        }
        for (var i = 0; i < retValue.length; i++) {
          var dim = dims[i];
          var val = retValue[i];
          var rawExtentOnDim = rawExtent[dim];
          var dimStore = targetChunks[dim];
          if (dimStore) {
            dimStore[rawIndex] = val;
          }
          if (val < rawExtentOnDim[0]) {
            rawExtentOnDim[0] = val;
          }
          if (val > rawExtentOnDim[1]) {
            rawExtentOnDim[1] = val;
          }
        }
      }
    }
  };
  /**
   * Large data down sampling using largest-triangle-three-buckets
   * @param {string} valueDimension
   * @param {number} targetCount
   */
  DataStore.prototype.lttbDownSample = function (valueDimension, rate) {
    var target = this.clone([valueDimension], true);
    var targetStorage = target._chunks;
    var dimStore = targetStorage[valueDimension];
    var len = this.count();
    var sampledIndex = 0;
    var frameSize = Math.floor(1 / rate);
    var currentRawIndex = this.getRawIndex(0);
    var maxArea;
    var area;
    var nextRawIndex;
    var newIndices = new (getIndicesCtor(this._rawCount))(Math.min((Math.ceil(len / frameSize) + 2) * 2, len));
    // First frame use the first data.
    newIndices[sampledIndex++] = currentRawIndex;
    for (var i = 1; i < len - 1; i += frameSize) {
      var nextFrameStart = Math.min(i + frameSize, len - 1);
      var nextFrameEnd = Math.min(i + frameSize * 2, len);
      var avgX = (nextFrameEnd + nextFrameStart) / 2;
      var avgY = 0;
      for (var idx = nextFrameStart; idx < nextFrameEnd; idx++) {
        var rawIndex = this.getRawIndex(idx);
        var y = dimStore[rawIndex];
        if (isNaN(y)) {
          continue;
        }
        avgY += y;
      }
      avgY /= nextFrameEnd - nextFrameStart;
      var frameStart = i;
      var frameEnd = Math.min(i + frameSize, len);
      var pointAX = i - 1;
      var pointAY = dimStore[currentRawIndex];
      maxArea = -1;
      nextRawIndex = frameStart;
      var firstNaNIndex = -1;
      var countNaN = 0;
      // Find a point from current frame that construct a triangle with largest area with previous selected point
      // And the average of next frame.
      for (var idx = frameStart; idx < frameEnd; idx++) {
        var rawIndex = this.getRawIndex(idx);
        var y = dimStore[rawIndex];
        if (isNaN(y)) {
          countNaN++;
          if (firstNaNIndex < 0) {
            firstNaNIndex = rawIndex;
          }
          continue;
        }
        // Calculate triangle area over three buckets
        area = Math.abs((pointAX - avgX) * (y - pointAY) - (pointAX - idx) * (avgY - pointAY));
        if (area > maxArea) {
          maxArea = area;
          nextRawIndex = rawIndex; // Next a is this b
        }
      }
      if (countNaN > 0 && countNaN < frameEnd - frameStart) {
        // Append first NaN point in every bucket.
        // It is necessary to ensure the correct order of indices.
        newIndices[sampledIndex++] = Math.min(firstNaNIndex, nextRawIndex);
        nextRawIndex = Math.max(firstNaNIndex, nextRawIndex);
      }
      newIndices[sampledIndex++] = nextRawIndex;
      currentRawIndex = nextRawIndex; // This a is the next a (chosen b)
    }
    // First frame use the last data.
    newIndices[sampledIndex++] = this.getRawIndex(len - 1);
    target._count = sampledIndex;
    target._indices = newIndices;
    target.getRawIndex = this._getRawIdx;
    return target;
  };
  /**
   * Large data down sampling using min-max
   * @param {string} valueDimension
   * @param {number} rate
   */
  DataStore.prototype.minmaxDownSample = function (valueDimension, rate) {
    var target = this.clone([valueDimension], true);
    var targetStorage = target._chunks;
    var frameSize = Math.floor(1 / rate);
    var dimStore = targetStorage[valueDimension];
    var len = this.count();
    // Each frame results in 2 data points, one for min and one for max
    var newIndices = new (getIndicesCtor(this._rawCount))(Math.ceil(len / frameSize) * 2);
    var offset = 0;
    for (var i = 0; i < len; i += frameSize) {
      var minIndex = i;
      var minValue = dimStore[this.getRawIndex(minIndex)];
      var maxIndex = i;
      var maxValue = dimStore[this.getRawIndex(maxIndex)];
      var thisFrameSize = frameSize;
      // Handle final smaller frame
      if (i + frameSize > len) {
        thisFrameSize = len - i;
      }
      // Determine min and max within the current frame
      for (var k = 0; k < thisFrameSize; k++) {
        var rawIndex = this.getRawIndex(i + k);
        var value = dimStore[rawIndex];
        if (value < minValue) {
          minValue = value;
          minIndex = i + k;
        }
        if (value > maxValue) {
          maxValue = value;
          maxIndex = i + k;
        }
      }
      var rawMinIndex = this.getRawIndex(minIndex);
      var rawMaxIndex = this.getRawIndex(maxIndex);
      // Set the order of the min and max values, based on their ordering in the frame
      if (minIndex < maxIndex) {
        newIndices[offset++] = rawMinIndex;
        newIndices[offset++] = rawMaxIndex;
      } else {
        newIndices[offset++] = rawMaxIndex;
        newIndices[offset++] = rawMinIndex;
      }
    }
    target._count = offset;
    target._indices = newIndices;
    target._updateGetRawIdx();
    return target;
  };
  /**
   * Large data down sampling on given dimension
   * @param sampleIndex Sample index for name and id
   */
  DataStore.prototype.downSample = function (dimension, rate, sampleValue, sampleIndex) {
    var target = this.clone([dimension], true);
    var targetStorage = target._chunks;
    var frameValues = [];
    var frameSize = Math.floor(1 / rate);
    var dimStore = targetStorage[dimension];
    var len = this.count();
    var rawExtentOnDim = target._rawExtent[dimension] = getInitialExtent();
    var newIndices = new (getIndicesCtor(this._rawCount))(Math.ceil(len / frameSize));
    var offset = 0;
    for (var i = 0; i < len; i += frameSize) {
      // Last frame
      if (frameSize > len - i) {
        frameSize = len - i;
        frameValues.length = frameSize;
      }
      for (var k = 0; k < frameSize; k++) {
        var dataIdx = this.getRawIndex(i + k);
        frameValues[k] = dimStore[dataIdx];
      }
      var value = sampleValue(frameValues);
      var sampleFrameIdx = this.getRawIndex(Math.min(i + sampleIndex(frameValues, value) || 0, len - 1));
      // Only write value on the filtered data
      dimStore[sampleFrameIdx] = value;
      if (value < rawExtentOnDim[0]) {
        rawExtentOnDim[0] = value;
      }
      if (value > rawExtentOnDim[1]) {
        rawExtentOnDim[1] = value;
      }
      newIndices[offset++] = sampleFrameIdx;
    }
    target._count = offset;
    target._indices = newIndices;
    target._updateGetRawIdx();
    return target;
  };
  /**
   * Data iteration
   * @param ctx default this
   * @example
   *  list.each('x', function (x, idx) {});
   *  list.each(['x', 'y'], function (x, y, idx) {});
   *  list.each(function (idx) {})
   */
  DataStore.prototype.each = function (dims, cb) {
    if (!this._count) {
      return;
    }
    var dimSize = dims.length;
    var chunks = this._chunks;
    for (var i = 0, len = this.count(); i < len; i++) {
      var rawIdx = this.getRawIndex(i);
      // Simple optimization
      switch (dimSize) {
        case 0:
          cb(i);
          break;
        case 1:
          cb(chunks[dims[0]][rawIdx], i);
          break;
        case 2:
          cb(chunks[dims[0]][rawIdx], chunks[dims[1]][rawIdx], i);
          break;
        default:
          var k = 0;
          var value = [];
          for (; k < dimSize; k++) {
            value[k] = chunks[dims[k]][rawIdx];
          }
          // Index
          value[k] = i;
          cb.apply(null, value);
      }
    }
  };
  /**
   * Get extent of data in one dimension
   */
  DataStore.prototype.getDataExtent = function (dim) {
    // Make sure use concrete dim as cache name.
    var dimData = this._chunks[dim];
    var initialExtent = getInitialExtent();
    if (!dimData) {
      return initialExtent;
    }
    // Make more strict checkings to ensure hitting cache.
    var currEnd = this.count();
    // Consider the most cases when using data zoom, `getDataExtent`
    // happened before filtering. We cache raw extent, which is not
    // necessary to be cleared and recalculated when restore data.
    var useRaw = !this._indices;
    var dimExtent;
    if (useRaw) {
      return this._rawExtent[dim].slice();
    }
    dimExtent = this._extent[dim];
    if (dimExtent) {
      return dimExtent.slice();
    }
    dimExtent = initialExtent;
    var min = dimExtent[0];
    var max = dimExtent[1];
    for (var i = 0; i < currEnd; i++) {
      var rawIdx = this.getRawIndex(i);
      var value = dimData[rawIdx];
      value < min && (min = value);
      value > max && (max = value);
    }
    dimExtent = [min, max];
    this._extent[dim] = dimExtent;
    return dimExtent;
  };
  /**
   * Get raw data item
   */
  DataStore.prototype.getRawDataItem = function (idx) {
    var rawIdx = this.getRawIndex(idx);
    if (!this._provider.persistent) {
      var val = [];
      var chunks = this._chunks;
      for (var i = 0; i < chunks.length; i++) {
        val.push(chunks[i][rawIdx]);
      }
      return val;
    } else {
      return this._provider.getItem(rawIdx);
    }
  };
  /**
   * Clone shallow.
   *
   * @param clonedDims Determine which dims to clone. Will share the data if not specified.
   */
  DataStore.prototype.clone = function (clonedDims, ignoreIndices) {
    var target = new DataStore();
    var chunks = this._chunks;
    var clonedDimsMap = clonedDims && reduce(clonedDims, function (obj, dimIdx) {
      obj[dimIdx] = true;
      return obj;
    }, {});
    if (clonedDimsMap) {
      for (var i = 0; i < chunks.length; i++) {
        // Not clone if dim is not picked.
        target._chunks[i] = !clonedDimsMap[i] ? chunks[i] : cloneChunk(chunks[i]);
      }
    } else {
      target._chunks = chunks;
    }
    this._copyCommonProps(target);
    if (!ignoreIndices) {
      target._indices = this._cloneIndices();
    }
    target._updateGetRawIdx();
    return target;
  };
  DataStore.prototype._copyCommonProps = function (target) {
    target._count = this._count;
    target._rawCount = this._rawCount;
    target._provider = this._provider;
    target._dimensions = this._dimensions;
    target._extent = clone(this._extent);
    target._rawExtent = clone(this._rawExtent);
  };
  DataStore.prototype._cloneIndices = function () {
    if (this._indices) {
      var Ctor = this._indices.constructor;
      var indices = void 0;
      if (Ctor === Array) {
        var thisCount = this._indices.length;
        indices = new Ctor(thisCount);
        for (var i = 0; i < thisCount; i++) {
          indices[i] = this._indices[i];
        }
      } else {
        indices = new Ctor(this._indices);
      }
      return indices;
    }
    return null;
  };
  DataStore.prototype._getRawIdxIdentity = function (idx) {
    return idx;
  };
  DataStore.prototype._getRawIdx = function (idx) {
    if (idx < this._count && idx >= 0) {
      return this._indices[idx];
    }
    return -1;
  };
  DataStore.prototype._updateGetRawIdx = function () {
    this.getRawIndex = this._indices ? this._getRawIdx : this._getRawIdxIdentity;
  };
  DataStore.internalField = function () {
    function getDimValueSimply(dataItem, property, dataIndex, dimIndex) {
      return parseDataValue(dataItem[dimIndex], this._dimensions[dimIndex]);
    }
    defaultDimValueGetters = {
      arrayRows: getDimValueSimply,
      objectRows: function (dataItem, property, dataIndex, dimIndex) {
        return parseDataValue(dataItem[property], this._dimensions[dimIndex]);
      },
      keyedColumns: getDimValueSimply,
      original: function (dataItem, property, dataIndex, dimIndex) {
        // Performance sensitive, do not use modelUtil.getDataItemValue.
        // If dataItem is an plain object with no value field, the let `value`
        // will be assigned with the object, but it will be tread correctly
        // in the `convertValue`.
        var value = dataItem && (dataItem.value == null ? dataItem : dataItem.value);
        return parseDataValue(value instanceof Array ? value[dimIndex]
        // If value is a single number or something else not array.
        : value, this._dimensions[dimIndex]);
      },
      typedArray: function (dataItem, property, dataIndex, dimIndex) {
        return dataItem[dimIndex];
      }
    };
  }();
  return DataStore;
}();
export default DataStore;