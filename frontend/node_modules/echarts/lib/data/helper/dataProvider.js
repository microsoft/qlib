
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
var _a, _b, _c;
// TODO
// ??? refactor? check the outer usage of data provider.
// merge with defaultDimValueGetter?
import { isTypedArray, extend, assert, each, isObject, bind } from 'zrender/lib/core/util.js';
import { getDataItemValue } from '../../util/model.js';
import { createSourceFromSeriesDataOption, isSourceInstance } from '../Source.js';
import { SOURCE_FORMAT_ORIGINAL, SOURCE_FORMAT_OBJECT_ROWS, SOURCE_FORMAT_KEYED_COLUMNS, SOURCE_FORMAT_TYPED_ARRAY, SOURCE_FORMAT_ARRAY_ROWS, SERIES_LAYOUT_BY_COLUMN, SERIES_LAYOUT_BY_ROW } from '../../util/types.js';
var providerMethods;
var mountMethods;
/**
 * If normal array used, mutable chunk size is supported.
 * If typed array used, chunk size must be fixed.
 */
var DefaultDataProvider = /** @class */function () {
  function DefaultDataProvider(sourceParam, dimSize) {
    // let source: Source;
    var source = !isSourceInstance(sourceParam) ? createSourceFromSeriesDataOption(sourceParam) : sourceParam;
    // declare source is Source;
    this._source = source;
    var data = this._data = source.data;
    // Typed array. TODO IE10+?
    if (source.sourceFormat === SOURCE_FORMAT_TYPED_ARRAY) {
      if (process.env.NODE_ENV !== 'production') {
        if (dimSize == null) {
          throw new Error('Typed array data must specify dimension size');
        }
      }
      this._offset = 0;
      this._dimSize = dimSize;
      this._data = data;
    }
    mountMethods(this, data, source);
  }
  DefaultDataProvider.prototype.getSource = function () {
    return this._source;
  };
  DefaultDataProvider.prototype.count = function () {
    return 0;
  };
  DefaultDataProvider.prototype.getItem = function (idx, out) {
    return;
  };
  DefaultDataProvider.prototype.appendData = function (newData) {};
  DefaultDataProvider.prototype.clean = function () {};
  DefaultDataProvider.protoInitialize = function () {
    // PENDING: To avoid potential incompat (e.g., prototype
    // is visited somewhere), still init them on prototype.
    var proto = DefaultDataProvider.prototype;
    proto.pure = false;
    proto.persistent = true;
  }();
  DefaultDataProvider.internalField = function () {
    var _a;
    mountMethods = function (provider, data, source) {
      var sourceFormat = source.sourceFormat;
      var seriesLayoutBy = source.seriesLayoutBy;
      var startIndex = source.startIndex;
      var dimsDef = source.dimensionsDefine;
      var methods = providerMethods[getMethodMapKey(sourceFormat, seriesLayoutBy)];
      if (process.env.NODE_ENV !== 'production') {
        assert(methods, 'Invalide sourceFormat: ' + sourceFormat);
      }
      extend(provider, methods);
      if (sourceFormat === SOURCE_FORMAT_TYPED_ARRAY) {
        provider.getItem = getItemForTypedArray;
        provider.count = countForTypedArray;
        provider.fillStorage = fillStorageForTypedArray;
      } else {
        var rawItemGetter = getRawSourceItemGetter(sourceFormat, seriesLayoutBy);
        provider.getItem = bind(rawItemGetter, null, data, startIndex, dimsDef);
        var rawCounter = getRawSourceDataCounter(sourceFormat, seriesLayoutBy);
        provider.count = bind(rawCounter, null, data, startIndex, dimsDef);
      }
    };
    var getItemForTypedArray = function (idx, out) {
      idx = idx - this._offset;
      out = out || [];
      var data = this._data;
      var dimSize = this._dimSize;
      var offset = dimSize * idx;
      for (var i = 0; i < dimSize; i++) {
        out[i] = data[offset + i];
      }
      return out;
    };
    var fillStorageForTypedArray = function (start, end, storage, extent) {
      var data = this._data;
      var dimSize = this._dimSize;
      for (var dim = 0; dim < dimSize; dim++) {
        var dimExtent = extent[dim];
        var min = dimExtent[0] == null ? Infinity : dimExtent[0];
        var max = dimExtent[1] == null ? -Infinity : dimExtent[1];
        var count = end - start;
        var arr = storage[dim];
        for (var i = 0; i < count; i++) {
          // appendData with TypedArray will always do replace in provider.
          var val = data[i * dimSize + dim];
          arr[start + i] = val;
          val < min && (min = val);
          val > max && (max = val);
        }
        dimExtent[0] = min;
        dimExtent[1] = max;
      }
    };
    var countForTypedArray = function () {
      return this._data ? this._data.length / this._dimSize : 0;
    };
    providerMethods = (_a = {}, _a[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_COLUMN] = {
      pure: true,
      appendData: appendDataSimply
    }, _a[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_ROW] = {
      pure: true,
      appendData: function () {
        throw new Error('Do not support appendData when set seriesLayoutBy: "row".');
      }
    }, _a[SOURCE_FORMAT_OBJECT_ROWS] = {
      pure: true,
      appendData: appendDataSimply
    }, _a[SOURCE_FORMAT_KEYED_COLUMNS] = {
      pure: true,
      appendData: function (newData) {
        var data = this._data;
        each(newData, function (newCol, key) {
          var oldCol = data[key] || (data[key] = []);
          for (var i = 0; i < (newCol || []).length; i++) {
            oldCol.push(newCol[i]);
          }
        });
      }
    }, _a[SOURCE_FORMAT_ORIGINAL] = {
      appendData: appendDataSimply
    }, _a[SOURCE_FORMAT_TYPED_ARRAY] = {
      persistent: false,
      pure: true,
      appendData: function (newData) {
        if (process.env.NODE_ENV !== 'production') {
          assert(isTypedArray(newData), 'Added data must be TypedArray if data in initialization is TypedArray');
        }
        this._data = newData;
      },
      // Clean self if data is already used.
      clean: function () {
        // PENDING
        this._offset += this.count();
        this._data = null;
      }
    }, _a);
    function appendDataSimply(newData) {
      for (var i = 0; i < newData.length; i++) {
        this._data.push(newData[i]);
      }
    }
  }();
  return DefaultDataProvider;
}();
export { DefaultDataProvider };
var getItemSimply = function (rawData, startIndex, dimsDef, idx) {
  return rawData[idx];
};
var rawSourceItemGetterMap = (_a = {}, _a[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_COLUMN] = function (rawData, startIndex, dimsDef, idx) {
  return rawData[idx + startIndex];
}, _a[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_ROW] = function (rawData, startIndex, dimsDef, idx, out) {
  idx += startIndex;
  var item = out || [];
  var data = rawData;
  for (var i = 0; i < data.length; i++) {
    var row = data[i];
    item[i] = row ? row[idx] : null;
  }
  return item;
}, _a[SOURCE_FORMAT_OBJECT_ROWS] = getItemSimply, _a[SOURCE_FORMAT_KEYED_COLUMNS] = function (rawData, startIndex, dimsDef, idx, out) {
  var item = out || [];
  for (var i = 0; i < dimsDef.length; i++) {
    var dimName = dimsDef[i].name;
    if (process.env.NODE_ENV !== 'production') {
      if (dimName == null) {
        throw new Error();
      }
    }
    var col = rawData[dimName];
    item[i] = col ? col[idx] : null;
  }
  return item;
}, _a[SOURCE_FORMAT_ORIGINAL] = getItemSimply, _a);
export function getRawSourceItemGetter(sourceFormat, seriesLayoutBy) {
  var method = rawSourceItemGetterMap[getMethodMapKey(sourceFormat, seriesLayoutBy)];
  if (process.env.NODE_ENV !== 'production') {
    assert(method, 'Do not support get item on "' + sourceFormat + '", "' + seriesLayoutBy + '".');
  }
  return method;
}
var countSimply = function (rawData, startIndex, dimsDef) {
  return rawData.length;
};
var rawSourceDataCounterMap = (_b = {}, _b[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_COLUMN] = function (rawData, startIndex, dimsDef) {
  return Math.max(0, rawData.length - startIndex);
}, _b[SOURCE_FORMAT_ARRAY_ROWS + '_' + SERIES_LAYOUT_BY_ROW] = function (rawData, startIndex, dimsDef) {
  var row = rawData[0];
  return row ? Math.max(0, row.length - startIndex) : 0;
}, _b[SOURCE_FORMAT_OBJECT_ROWS] = countSimply, _b[SOURCE_FORMAT_KEYED_COLUMNS] = function (rawData, startIndex, dimsDef) {
  var dimName = dimsDef[0].name;
  if (process.env.NODE_ENV !== 'production') {
    if (dimName == null) {
      throw new Error();
    }
  }
  var col = rawData[dimName];
  return col ? col.length : 0;
}, _b[SOURCE_FORMAT_ORIGINAL] = countSimply, _b);
export function getRawSourceDataCounter(sourceFormat, seriesLayoutBy) {
  var method = rawSourceDataCounterMap[getMethodMapKey(sourceFormat, seriesLayoutBy)];
  if (process.env.NODE_ENV !== 'production') {
    assert(method, 'Do not support count on "' + sourceFormat + '", "' + seriesLayoutBy + '".');
  }
  return method;
}
var getRawValueSimply = function (dataItem, dimIndex, property) {
  return dataItem[dimIndex];
};
var rawSourceValueGetterMap = (_c = {}, _c[SOURCE_FORMAT_ARRAY_ROWS] = getRawValueSimply, _c[SOURCE_FORMAT_OBJECT_ROWS] = function (dataItem, dimIndex, property) {
  return dataItem[property];
}, _c[SOURCE_FORMAT_KEYED_COLUMNS] = getRawValueSimply, _c[SOURCE_FORMAT_ORIGINAL] = function (dataItem, dimIndex, property) {
  // FIXME: In some case (markpoint in geo (geo-map.html)),
  // dataItem is {coord: [...]}
  var value = getDataItemValue(dataItem);
  return !(value instanceof Array) ? value : value[dimIndex];
}, _c[SOURCE_FORMAT_TYPED_ARRAY] = getRawValueSimply, _c);
export function getRawSourceValueGetter(sourceFormat) {
  var method = rawSourceValueGetterMap[sourceFormat];
  if (process.env.NODE_ENV !== 'production') {
    assert(method, 'Do not support get value on "' + sourceFormat + '".');
  }
  return method;
}
function getMethodMapKey(sourceFormat, seriesLayoutBy) {
  return sourceFormat === SOURCE_FORMAT_ARRAY_ROWS ? sourceFormat + '_' + seriesLayoutBy : sourceFormat;
}
// ??? FIXME can these logic be more neat: getRawValue, getRawDataItem,
// Consider persistent.
// Caution: why use raw value to display on label or tooltip?
// A reason is to avoid format. For example time value we do not know
// how to format is expected. More over, if stack is used, calculated
// value may be 0.91000000001, which have brings trouble to display.
// TODO: consider how to treat null/undefined/NaN when display?
export function retrieveRawValue(data, dataIndex,
// If dimIndex is null/undefined, return OptionDataItem.
// Otherwise, return OptionDataValue.
dim) {
  if (!data) {
    return;
  }
  // Consider data may be not persistent.
  var dataItem = data.getRawDataItem(dataIndex);
  if (dataItem == null) {
    return;
  }
  var store = data.getStore();
  var sourceFormat = store.getSource().sourceFormat;
  if (dim != null) {
    var dimIndex = data.getDimensionIndex(dim);
    var property = store.getDimensionProperty(dimIndex);
    return getRawSourceValueGetter(sourceFormat)(dataItem, dimIndex, property);
  } else {
    var result = dataItem;
    if (sourceFormat === SOURCE_FORMAT_ORIGINAL) {
      result = getDataItemValue(dataItem);
    }
    return result;
  }
}
/**
 * Compatible with some cases (in pie, map) like:
 * data: [{name: 'xx', value: 5, selected: true}, ...]
 * where only sourceFormat is 'original' and 'objectRows' supported.
 *
 * // TODO
 * Supported detail options in data item when using 'arrayRows'.
 *
 * @param data
 * @param dataIndex
 * @param attr like 'selected'
 */
export function retrieveRawAttr(data, dataIndex, attr) {
  if (!data) {
    return;
  }
  var sourceFormat = data.getStore().getSource().sourceFormat;
  if (sourceFormat !== SOURCE_FORMAT_ORIGINAL && sourceFormat !== SOURCE_FORMAT_OBJECT_ROWS) {
    return;
  }
  var dataItem = data.getRawDataItem(dataIndex);
  if (sourceFormat === SOURCE_FORMAT_ORIGINAL && !isObject(dataItem)) {
    dataItem = null;
  }
  if (dataItem) {
    return dataItem[attr];
  }
}