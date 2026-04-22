
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
import { SERIES_LAYOUT_BY_COLUMN, SOURCE_FORMAT_OBJECT_ROWS, SOURCE_FORMAT_ARRAY_ROWS } from '../../util/types.js';
import { normalizeToArray } from '../../util/model.js';
import { createHashMap, bind, each, hasOwn, map, clone, isObject, extend, isNumber } from 'zrender/lib/core/util.js';
import { getRawSourceItemGetter, getRawSourceDataCounter, getRawSourceValueGetter } from './dataProvider.js';
import { parseDataValue } from './dataValueHelper.js';
import { log, makePrintable, throwError } from '../../util/log.js';
import { createSource, detectSourceFormat } from '../Source.js';
/**
 * TODO: disable writable.
 * This structure will be exposed to users.
 */
var ExternalSource = /** @class */function () {
  function ExternalSource() {}
  ExternalSource.prototype.getRawData = function () {
    // Only built-in transform available.
    throw new Error('not supported');
  };
  ExternalSource.prototype.getRawDataItem = function (dataIndex) {
    // Only built-in transform available.
    throw new Error('not supported');
  };
  ExternalSource.prototype.cloneRawData = function () {
    return;
  };
  /**
   * @return If dimension not found, return null/undefined.
   */
  ExternalSource.prototype.getDimensionInfo = function (dim) {
    return;
  };
  /**
   * dimensions defined if and only if either:
   * (a) dataset.dimensions are declared.
   * (b) dataset data include dimensions definitions in data (detected or via specified `sourceHeader`).
   * If dimensions are defined, `dimensionInfoAll` is corresponding to
   * the defined dimensions.
   * Otherwise, `dimensionInfoAll` is determined by data columns.
   * @return Always return an array (even empty array).
   */
  ExternalSource.prototype.cloneAllDimensionInfo = function () {
    return;
  };
  ExternalSource.prototype.count = function () {
    return;
  };
  /**
   * Only support by dimension index.
   * No need to support by dimension name in transform function,
   * because transform function is not case-specific, no need to use name literally.
   */
  ExternalSource.prototype.retrieveValue = function (dataIndex, dimIndex) {
    return;
  };
  ExternalSource.prototype.retrieveValueFromItem = function (dataItem, dimIndex) {
    return;
  };
  ExternalSource.prototype.convertValue = function (rawVal, dimInfo) {
    return parseDataValue(rawVal, dimInfo);
  };
  return ExternalSource;
}();
export { ExternalSource };
function createExternalSource(internalSource, externalTransform) {
  var extSource = new ExternalSource();
  var data = internalSource.data;
  var sourceFormat = extSource.sourceFormat = internalSource.sourceFormat;
  var sourceHeaderCount = internalSource.startIndex;
  var errMsg = '';
  if (internalSource.seriesLayoutBy !== SERIES_LAYOUT_BY_COLUMN) {
    // For the logic simplicity in transformer, only 'culumn' is
    // supported in data transform. Otherwise, the `dimensionsDefine`
    // might be detected by 'row', which probably confuses users.
    if (process.env.NODE_ENV !== 'production') {
      errMsg = '`seriesLayoutBy` of upstream dataset can only be "column" in data transform.';
    }
    throwError(errMsg);
  }
  // [MEMO]
  // Create a new dimensions structure for exposing.
  // Do not expose all dimension info to users directly.
  // Because the dimension is probably auto detected from data and not might reliable.
  // Should not lead the transformers to think that is reliable and return it.
  // See [DIMENSION_INHERIT_RULE] in `sourceManager.ts`.
  var dimensions = [];
  var dimsByName = {};
  var dimsDef = internalSource.dimensionsDefine;
  if (dimsDef) {
    each(dimsDef, function (dimDef, idx) {
      var name = dimDef.name;
      var dimDefExt = {
        index: idx,
        name: name,
        displayName: dimDef.displayName
      };
      dimensions.push(dimDefExt);
      // Users probably do not specify dimension name. For simplicity, data transform
      // does not generate dimension name.
      if (name != null) {
        // Dimension name should not be duplicated.
        // For simplicity, data transform forbids name duplication, do not generate
        // new name like module `completeDimensions.ts` did, but just tell users.
        var errMsg_1 = '';
        if (hasOwn(dimsByName, name)) {
          if (process.env.NODE_ENV !== 'production') {
            errMsg_1 = 'dimension name "' + name + '" duplicated.';
          }
          throwError(errMsg_1);
        }
        dimsByName[name] = dimDefExt;
      }
    });
  }
  // If dimension definitions are not defined and can not be detected.
  // e.g., pure data `[[11, 22], ...]`.
  else {
    for (var i = 0; i < internalSource.dimensionsDetectedCount || 0; i++) {
      // Do not generete name or anything others. The consequence process in
      // `transform` or `series` probably have there own name generation strategry.
      dimensions.push({
        index: i
      });
    }
  }
  // Implement public methods:
  var rawItemGetter = getRawSourceItemGetter(sourceFormat, SERIES_LAYOUT_BY_COLUMN);
  if (externalTransform.__isBuiltIn) {
    extSource.getRawDataItem = function (dataIndex) {
      return rawItemGetter(data, sourceHeaderCount, dimensions, dataIndex);
    };
    extSource.getRawData = bind(getRawData, null, internalSource);
  }
  extSource.cloneRawData = bind(cloneRawData, null, internalSource);
  var rawCounter = getRawSourceDataCounter(sourceFormat, SERIES_LAYOUT_BY_COLUMN);
  extSource.count = bind(rawCounter, null, data, sourceHeaderCount, dimensions);
  var rawValueGetter = getRawSourceValueGetter(sourceFormat);
  extSource.retrieveValue = function (dataIndex, dimIndex) {
    var rawItem = rawItemGetter(data, sourceHeaderCount, dimensions, dataIndex);
    return retrieveValueFromItem(rawItem, dimIndex);
  };
  var retrieveValueFromItem = extSource.retrieveValueFromItem = function (dataItem, dimIndex) {
    if (dataItem == null) {
      return;
    }
    var dimDef = dimensions[dimIndex];
    // When `dimIndex` is `null`, `rawValueGetter` return the whole item.
    if (dimDef) {
      return rawValueGetter(dataItem, dimIndex, dimDef.name);
    }
  };
  extSource.getDimensionInfo = bind(getDimensionInfo, null, dimensions, dimsByName);
  extSource.cloneAllDimensionInfo = bind(cloneAllDimensionInfo, null, dimensions);
  return extSource;
}
function getRawData(upstream) {
  var sourceFormat = upstream.sourceFormat;
  if (!isSupportedSourceFormat(sourceFormat)) {
    var errMsg = '';
    if (process.env.NODE_ENV !== 'production') {
      errMsg = '`getRawData` is not supported in source format ' + sourceFormat;
    }
    throwError(errMsg);
  }
  return upstream.data;
}
function cloneRawData(upstream) {
  var sourceFormat = upstream.sourceFormat;
  var data = upstream.data;
  if (!isSupportedSourceFormat(sourceFormat)) {
    var errMsg = '';
    if (process.env.NODE_ENV !== 'production') {
      errMsg = '`cloneRawData` is not supported in source format ' + sourceFormat;
    }
    throwError(errMsg);
  }
  if (sourceFormat === SOURCE_FORMAT_ARRAY_ROWS) {
    var result = [];
    for (var i = 0, len = data.length; i < len; i++) {
      // Not strictly clone for performance
      result.push(data[i].slice());
    }
    return result;
  } else if (sourceFormat === SOURCE_FORMAT_OBJECT_ROWS) {
    var result = [];
    for (var i = 0, len = data.length; i < len; i++) {
      // Not strictly clone for performance
      result.push(extend({}, data[i]));
    }
    return result;
  }
}
function getDimensionInfo(dimensions, dimsByName, dim) {
  if (dim == null) {
    return;
  }
  // Keep the same logic as `List::getDimension` did.
  if (isNumber(dim)
  // If being a number-like string but not being defined a dimension name.
  || !isNaN(dim) && !hasOwn(dimsByName, dim)) {
    return dimensions[dim];
  } else if (hasOwn(dimsByName, dim)) {
    return dimsByName[dim];
  }
}
function cloneAllDimensionInfo(dimensions) {
  return clone(dimensions);
}
var externalTransformMap = createHashMap();
export function registerExternalTransform(externalTransform) {
  externalTransform = clone(externalTransform);
  var type = externalTransform.type;
  var errMsg = '';
  if (!type) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'Must have a `type` when `registerTransform`.';
    }
    throwError(errMsg);
  }
  var typeParsed = type.split(':');
  if (typeParsed.length !== 2) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'Name must include namespace like "ns:regression".';
    }
    throwError(errMsg);
  }
  // Namespace 'echarts:xxx' is official namespace, where the transforms should
  // be called directly via 'xxx' rather than 'echarts:xxx'.
  var isBuiltIn = false;
  if (typeParsed[0] === 'echarts') {
    type = typeParsed[1];
    isBuiltIn = true;
  }
  externalTransform.__isBuiltIn = isBuiltIn;
  externalTransformMap.set(type, externalTransform);
}
export function applyDataTransform(rawTransOption, sourceList, infoForPrint) {
  var pipedTransOption = normalizeToArray(rawTransOption);
  var pipeLen = pipedTransOption.length;
  var errMsg = '';
  if (!pipeLen) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'If `transform` declared, it should at least contain one transform.';
    }
    throwError(errMsg);
  }
  for (var i = 0, len = pipeLen; i < len; i++) {
    var transOption = pipedTransOption[i];
    sourceList = applySingleDataTransform(transOption, sourceList, infoForPrint, pipeLen === 1 ? null : i);
    // piped transform only support single input, except the fist one.
    // piped transform only support single output, except the last one.
    if (i !== len - 1) {
      sourceList.length = Math.max(sourceList.length, 1);
    }
  }
  return sourceList;
}
function applySingleDataTransform(transOption, upSourceList, infoForPrint,
// If `pipeIndex` is null/undefined, no piped transform.
pipeIndex) {
  var errMsg = '';
  if (!upSourceList.length) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'Must have at least one upstream dataset.';
    }
    throwError(errMsg);
  }
  if (!isObject(transOption)) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'transform declaration must be an object rather than ' + typeof transOption + '.';
    }
    throwError(errMsg);
  }
  var transType = transOption.type;
  var externalTransform = externalTransformMap.get(transType);
  if (!externalTransform) {
    if (process.env.NODE_ENV !== 'production') {
      errMsg = 'Can not find transform on type "' + transType + '".';
    }
    throwError(errMsg);
  }
  // Prepare source
  var extUpSourceList = map(upSourceList, function (upSource) {
    return createExternalSource(upSource, externalTransform);
  });
  var resultList = normalizeToArray(externalTransform.transform({
    upstream: extUpSourceList[0],
    upstreamList: extUpSourceList,
    config: clone(transOption.config)
  }));
  if (process.env.NODE_ENV !== 'production') {
    if (transOption.print) {
      var printStrArr = map(resultList, function (extSource) {
        var pipeIndexStr = pipeIndex != null ? ' === pipe index: ' + pipeIndex : '';
        return ['=== dataset index: ' + infoForPrint.datasetIndex + pipeIndexStr + ' ===', '- transform result data:', makePrintable(extSource.data), '- transform result dimensions:', makePrintable(extSource.dimensions)].join('\n');
      }).join('\n');
      log(printStrArr);
    }
  }
  return map(resultList, function (result, resultIndex) {
    var errMsg = '';
    if (!isObject(result)) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'A transform should not return some empty results.';
      }
      throwError(errMsg);
    }
    if (!result.data) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'Transform result data should be not be null or undefined';
      }
      throwError(errMsg);
    }
    var sourceFormat = detectSourceFormat(result.data);
    if (!isSupportedSourceFormat(sourceFormat)) {
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'Transform result data should be array rows or object rows.';
      }
      throwError(errMsg);
    }
    var resultMetaRawOption;
    var firstUpSource = upSourceList[0];
    /**
     * Intuitively, the end users known the content of the original `dataset.source`,
     * calucating the transform result in mind.
     * Suppose the original `dataset.source` is:
     * ```js
     * [
     *     ['product', '2012', '2013', '2014', '2015'],
     *     ['AAA', 41.1, 30.4, 65.1, 53.3],
     *     ['BBB', 86.5, 92.1, 85.7, 83.1],
     *     ['CCC', 24.1, 67.2, 79.5, 86.4]
     * ]
     * ```
     * The dimension info have to be detected from the source data.
     * Some of the transformers (like filter, sort) will follow the dimension info
     * of upstream, while others use new dimensions (like aggregate).
     * Transformer can output a field `dimensions` to define the its own output dimensions.
     * We also allow transformers to ignore the output `dimensions` field, and
     * inherit the upstream dimensions definition. It can reduce the burden of handling
     * dimensions in transformers.
     *
     * See also [DIMENSION_INHERIT_RULE] in `sourceManager.ts`.
     */
    if (firstUpSource && resultIndex === 0
    // If transformer returns `dimensions`, it means that the transformer has different
    // dimensions definitions. We do not inherit anything from upstream.
    && !result.dimensions) {
      var startIndex = firstUpSource.startIndex;
      // We copy the header of upstream to the result, because:
      // (1) The returned data always does not contain header line and can not be used
      // as dimension-detection. In this case we can not use "detected dimensions" of
      // upstream directly, because it might be detected based on different `seriesLayoutBy`.
      // (2) We should support that the series read the upstream source in `seriesLayoutBy: 'row'`.
      // So the original detected header should be add to the result, otherwise they can not be read.
      if (startIndex) {
        result.data = firstUpSource.data.slice(0, startIndex).concat(result.data);
      }
      resultMetaRawOption = {
        seriesLayoutBy: SERIES_LAYOUT_BY_COLUMN,
        sourceHeader: startIndex,
        dimensions: firstUpSource.metaRawOption.dimensions
      };
    } else {
      resultMetaRawOption = {
        seriesLayoutBy: SERIES_LAYOUT_BY_COLUMN,
        sourceHeader: 0,
        dimensions: result.dimensions
      };
    }
    return createSource(result.data, resultMetaRawOption, null);
  });
}
function isSupportedSourceFormat(sourceFormat) {
  return sourceFormat === SOURCE_FORMAT_ARRAY_ROWS || sourceFormat === SOURCE_FORMAT_OBJECT_ROWS;
}