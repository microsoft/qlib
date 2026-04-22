
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
import { __extends } from "tslib";
import * as zrUtil from 'zrender/lib/core/util.js';
import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import View from '../View.js';
import geoSourceManager from './geoSourceManager.js';
import { SINGLE_REFERRING } from '../../util/model.js';
import { warn } from '../../util/log.js';
var GEO_DEFAULT_PARAMS = {
  'geoJSON': {
    aspectScale: 0.75,
    invertLongitute: true
  },
  'geoSVG': {
    aspectScale: 1,
    invertLongitute: false
  }
};
export var geo2DDimensions = ['lng', 'lat'];
var Geo = /** @class */function (_super) {
  __extends(Geo, _super);
  function Geo(name, map, opt) {
    var _this = _super.call(this, name) || this;
    _this.dimensions = geo2DDimensions;
    _this.type = 'geo';
    // Only store specified name coord via `addGeoCoord`.
    _this._nameCoordMap = zrUtil.createHashMap();
    _this.map = map;
    var projection = opt.projection;
    var source = geoSourceManager.load(map, opt.nameMap, opt.nameProperty);
    var resource = geoSourceManager.getGeoResource(map);
    var resourceType = _this.resourceType = resource ? resource.type : null;
    var regions = _this.regions = source.regions;
    var defaultParams = GEO_DEFAULT_PARAMS[resource.type];
    _this._regionsMap = source.regionsMap;
    _this.regions = source.regions;
    if (process.env.NODE_ENV !== 'production' && projection) {
      // Do some check
      if (resourceType === 'geoSVG') {
        if (process.env.NODE_ENV !== 'production') {
          warn("Map " + map + " with SVG source can't use projection. Only GeoJSON source supports projection.");
        }
        projection = null;
      }
      if (!(projection.project && projection.unproject)) {
        if (process.env.NODE_ENV !== 'production') {
          warn('project and unproject must be both provided in the projeciton.');
        }
        projection = null;
      }
    }
    _this.projection = projection;
    var boundingRect;
    if (projection) {
      // Can't reuse the raw bounding rect
      for (var i = 0; i < regions.length; i++) {
        var regionRect = regions[i].getBoundingRect(projection);
        boundingRect = boundingRect || regionRect.clone();
        boundingRect.union(regionRect);
      }
    } else {
      boundingRect = source.boundingRect;
    }
    _this.setBoundingRect(boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height);
    // aspectScale and invertLongitute actually is the parameters default raw projection.
    // So we ignore them if projection is given.
    // Ignore default aspect scale if projection exits.
    _this.aspectScale = projection ? 1 : zrUtil.retrieve2(opt.aspectScale, defaultParams.aspectScale);
    // Not invert longitude if projection exits.
    _this._invertLongitute = projection ? false : defaultParams.invertLongitute;
    return _this;
  }
  Geo.prototype._transformTo = function (x, y, width, height) {
    var rect = this.getBoundingRect();
    var invertLongitute = this._invertLongitute;
    rect = rect.clone();
    if (invertLongitute) {
      // Longitude is inverted.
      rect.y = -rect.y - rect.height;
    }
    var rawTransformable = this._rawTransformable;
    rawTransformable.transform = rect.calculateTransform(new BoundingRect(x, y, width, height));
    var rawParent = rawTransformable.parent;
    rawTransformable.parent = null;
    rawTransformable.decomposeTransform();
    rawTransformable.parent = rawParent;
    if (invertLongitute) {
      rawTransformable.scaleY = -rawTransformable.scaleY;
    }
    this._updateTransform();
  };
  Geo.prototype.getRegion = function (name) {
    return this._regionsMap.get(name);
  };
  Geo.prototype.getRegionByCoord = function (coord) {
    var regions = this.regions;
    for (var i = 0; i < regions.length; i++) {
      var region = regions[i];
      if (region.type === 'geoJSON' && region.contain(coord)) {
        return regions[i];
      }
    }
  };
  /**
   * Add geoCoord for indexing by name
   */
  Geo.prototype.addGeoCoord = function (name, geoCoord) {
    this._nameCoordMap.set(name, geoCoord);
  };
  /**
   * Get geoCoord by name
   */
  Geo.prototype.getGeoCoord = function (name) {
    var region = this._regionsMap.get(name);
    // Calculate center only on demand.
    return this._nameCoordMap.get(name) || region && region.getCenter();
  };
  Geo.prototype.dataToPoint = function (data, noRoam, out) {
    if (zrUtil.isString(data)) {
      // Map area name to geoCoord
      data = this.getGeoCoord(data);
    }
    if (data) {
      var projection = this.projection;
      if (projection) {
        // projection may return null point.
        data = projection.project(data);
      }
      return data && this.projectedToPoint(data, noRoam, out);
    }
  };
  Geo.prototype.pointToData = function (point) {
    var projection = this.projection;
    if (projection) {
      // projection may return null point.
      point = projection.unproject(point);
    }
    return point && this.pointToProjected(point);
  };
  /**
   * Point to projected data. Same with pointToData when projection is used.
   */
  Geo.prototype.pointToProjected = function (point) {
    return _super.prototype.pointToData.call(this, point);
  };
  Geo.prototype.projectedToPoint = function (projected, noRoam, out) {
    return _super.prototype.dataToPoint.call(this, projected, noRoam, out);
  };
  Geo.prototype.convertToPixel = function (ecModel, finder, value) {
    var coordSys = getCoordSys(finder);
    return coordSys === this ? coordSys.dataToPoint(value) : null;
  };
  Geo.prototype.convertFromPixel = function (ecModel, finder, pixel) {
    var coordSys = getCoordSys(finder);
    return coordSys === this ? coordSys.pointToData(pixel) : null;
  };
  return Geo;
}(View);
;
zrUtil.mixin(Geo, View);
function getCoordSys(finder) {
  var geoModel = finder.geoModel;
  var seriesModel = finder.seriesModel;
  return geoModel ? geoModel.coordinateSystem : seriesModel ? seriesModel.coordinateSystem // For map series.
  || (seriesModel.getReferringComponents('geo', SINGLE_REFERRING).models[0] || {}).coordinateSystem : null;
}
export default Geo;