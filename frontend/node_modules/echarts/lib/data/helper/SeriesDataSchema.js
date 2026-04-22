
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
import { createHashMap, isObject, retrieve2 } from 'zrender/lib/core/util.js';
import { makeInner } from '../../util/model.js';
import { shouldRetrieveDataByName } from '../Source.js';
var inner = makeInner();
var dimTypeShort = {
  float: 'f',
  int: 'i',
  ordinal: 'o',
  number: 'n',
  time: 't'
};
/**
 * Represents the dimension requirement of a series.
 *
 * NOTICE:
 * When there are too many dimensions in dataset and many series, only the used dimensions
 * (i.e., used by coord sys and declared in `series.encode`) are add to `dimensionDefineList`.
 * But users may query data by other unused dimension names.
 * In this case, users can only query data if and only if they have defined dimension names
 * via ec option, so we provide `getDimensionIndexFromSource`, which only query them from
 * `source` dimensions.
 */
var SeriesDataSchema = /** @class */function () {
  function SeriesDataSchema(opt) {
    this.dimensions = opt.dimensions;
    this._dimOmitted = opt.dimensionOmitted;
    this.source = opt.source;
    this._fullDimCount = opt.fullDimensionCount;
    this._updateDimOmitted(opt.dimensionOmitted);
  }
  SeriesDataSchema.prototype.isDimensionOmitted = function () {
    return this._dimOmitted;
  };
  SeriesDataSchema.prototype._updateDimOmitted = function (dimensionOmitted) {
    this._dimOmitted = dimensionOmitted;
    if (!dimensionOmitted) {
      return;
    }
    if (!this._dimNameMap) {
      this._dimNameMap = ensureSourceDimNameMap(this.source);
    }
  };
  /**
   * @caution Can only be used when `dimensionOmitted: true`.
   *
   * Get index by user defined dimension name (i.e., not internal generate name).
   * That is, get index from `dimensionsDefine`.
   * If no `dimensionsDefine`, or no name get, return -1.
   */
  SeriesDataSchema.prototype.getSourceDimensionIndex = function (dimName) {
    return retrieve2(this._dimNameMap.get(dimName), -1);
  };
  /**
   * @caution Can only be used when `dimensionOmitted: true`.
   *
   * Notice: may return `null`/`undefined` if user not specify dimension names.
   */
  SeriesDataSchema.prototype.getSourceDimension = function (dimIndex) {
    var dimensionsDefine = this.source.dimensionsDefine;
    if (dimensionsDefine) {
      return dimensionsDefine[dimIndex];
    }
  };
  SeriesDataSchema.prototype.makeStoreSchema = function () {
    var dimCount = this._fullDimCount;
    var willRetrieveDataByName = shouldRetrieveDataByName(this.source);
    var makeHashStrict = !shouldOmitUnusedDimensions(dimCount);
    // If source don't have dimensions or series don't omit unsed dimensions.
    // Generate from seriesDimList directly
    var dimHash = '';
    var dims = [];
    for (var fullDimIdx = 0, seriesDimIdx = 0; fullDimIdx < dimCount; fullDimIdx++) {
      var property = void 0;
      var type = void 0;
      var ordinalMeta = void 0;
      var seriesDimDef = this.dimensions[seriesDimIdx];
      // The list has been sorted by `storeDimIndex` asc.
      if (seriesDimDef && seriesDimDef.storeDimIndex === fullDimIdx) {
        property = willRetrieveDataByName ? seriesDimDef.name : null;
        type = seriesDimDef.type;
        ordinalMeta = seriesDimDef.ordinalMeta;
        seriesDimIdx++;
      } else {
        var sourceDimDef = this.getSourceDimension(fullDimIdx);
        if (sourceDimDef) {
          property = willRetrieveDataByName ? sourceDimDef.name : null;
          type = sourceDimDef.type;
        }
      }
      dims.push({
        property: property,
        type: type,
        ordinalMeta: ordinalMeta
      });
      // If retrieving data by index,
      //   use <index, type, ordinalMeta> to determine whether data can be shared.
      //   (Because in this case there might be no dimension name defined in dataset, but indices always exists).
      //   (Indices are always 0, 1, 2, ..., so we can ignore them to shorten the hash).
      // Otherwise if retrieving data by property name (like `data: [{aa: 123, bb: 765}, ...]`),
      //   use <property, type, ordinalMeta> in hash.
      if (willRetrieveDataByName && property != null
      // For data stack, we have make sure each series has its own dim on this store.
      // So we do not add property to hash to make sure they can share this store.
      && (!seriesDimDef || !seriesDimDef.isCalculationCoord)) {
        dimHash += makeHashStrict
        // Use escape character '`' in case that property name contains '$'.
        ? property.replace(/\`/g, '`1').replace(/\$/g, '`2')
        // For better performance, when there are large dimensions, tolerant this defects that hardly meet.
        : property;
      }
      dimHash += '$';
      dimHash += dimTypeShort[type] || 'f';
      if (ordinalMeta) {
        dimHash += ordinalMeta.uid;
      }
      dimHash += '$';
    }
    // Source from endpoint(usually series) will be read differently
    // when seriesLayoutBy or startIndex(which is affected by sourceHeader) are different.
    // So we use this three props as key.
    var source = this.source;
    var hash = [source.seriesLayoutBy, source.startIndex, dimHash].join('$$');
    return {
      dimensions: dims,
      hash: hash
    };
  };
  SeriesDataSchema.prototype.makeOutputDimensionNames = function () {
    var result = [];
    for (var fullDimIdx = 0, seriesDimIdx = 0; fullDimIdx < this._fullDimCount; fullDimIdx++) {
      var name_1 = void 0;
      var seriesDimDef = this.dimensions[seriesDimIdx];
      // The list has been sorted by `storeDimIndex` asc.
      if (seriesDimDef && seriesDimDef.storeDimIndex === fullDimIdx) {
        if (!seriesDimDef.isCalculationCoord) {
          name_1 = seriesDimDef.name;
        }
        seriesDimIdx++;
      } else {
        var sourceDimDef = this.getSourceDimension(fullDimIdx);
        if (sourceDimDef) {
          name_1 = sourceDimDef.name;
        }
      }
      result.push(name_1);
    }
    return result;
  };
  SeriesDataSchema.prototype.appendCalculationDimension = function (dimDef) {
    this.dimensions.push(dimDef);
    dimDef.isCalculationCoord = true;
    this._fullDimCount++;
    // If append dimension on a data store, consider the store
    // might be shared by different series, series dimensions not
    // really map to store dimensions.
    this._updateDimOmitted(true);
  };
  return SeriesDataSchema;
}();
export { SeriesDataSchema };
export function isSeriesDataSchema(schema) {
  return schema instanceof SeriesDataSchema;
}
export function createDimNameMap(dimsDef) {
  var dataDimNameMap = createHashMap();
  for (var i = 0; i < (dimsDef || []).length; i++) {
    var dimDefItemRaw = dimsDef[i];
    var userDimName = isObject(dimDefItemRaw) ? dimDefItemRaw.name : dimDefItemRaw;
    if (userDimName != null && dataDimNameMap.get(userDimName) == null) {
      dataDimNameMap.set(userDimName, i);
    }
  }
  return dataDimNameMap;
}
export function ensureSourceDimNameMap(source) {
  var innerSource = inner(source);
  return innerSource.dimNameMap || (innerSource.dimNameMap = createDimNameMap(source.dimensionsDefine));
}
export function shouldOmitUnusedDimensions(dimCount) {
  return dimCount > 30;
}