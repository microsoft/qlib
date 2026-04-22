
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
import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import * as vec2 from 'zrender/lib/core/vector.js';
import * as polygonContain from 'zrender/lib/contain/polygon.js';
import * as matrix from 'zrender/lib/core/matrix.js';
import { each } from 'zrender/lib/core/util.js';
var TMP_TRANSFORM = [];
function transformPoints(points, transform) {
  for (var p = 0; p < points.length; p++) {
    vec2.applyTransform(points[p], points[p], transform);
  }
}
function updateBBoxFromPoints(points, min, max, projection) {
  for (var i = 0; i < points.length; i++) {
    var p = points[i];
    if (projection) {
      // projection may return null point.
      p = projection.project(p);
    }
    if (p && isFinite(p[0]) && isFinite(p[1])) {
      vec2.min(min, min, p);
      vec2.max(max, max, p);
    }
  }
}
function centroid(points) {
  var signedArea = 0;
  var cx = 0;
  var cy = 0;
  var len = points.length;
  var x0 = points[len - 1][0];
  var y0 = points[len - 1][1];
  // Polygon should been closed.
  for (var i = 0; i < len; i++) {
    var x1 = points[i][0];
    var y1 = points[i][1];
    var a = x0 * y1 - x1 * y0;
    signedArea += a;
    cx += (x0 + x1) * a;
    cy += (y0 + y1) * a;
    x0 = x1;
    y0 = y1;
  }
  return signedArea ? [cx / signedArea / 3, cy / signedArea / 3, signedArea] : [points[0][0] || 0, points[0][1] || 0];
}
var Region = /** @class */function () {
  function Region(name) {
    this.name = name;
  }
  Region.prototype.setCenter = function (center) {
    this._center = center;
  };
  /**
   * Get center point in data unit. That is,
   * for GeoJSONRegion, the unit is lat/lng,
   * for GeoSVGRegion, the unit is SVG local coord.
   */
  Region.prototype.getCenter = function () {
    var center = this._center;
    if (!center) {
      // In most cases there are no need to calculate this center.
      // So calculate only when called.
      center = this._center = this.calcCenter();
    }
    return center;
  };
  return Region;
}();
export { Region };
var GeoJSONPolygonGeometry = /** @class */function () {
  function GeoJSONPolygonGeometry(exterior, interiors) {
    this.type = 'polygon';
    this.exterior = exterior;
    this.interiors = interiors;
  }
  return GeoJSONPolygonGeometry;
}();
export { GeoJSONPolygonGeometry };
var GeoJSONLineStringGeometry = /** @class */function () {
  function GeoJSONLineStringGeometry(points) {
    this.type = 'linestring';
    this.points = points;
  }
  return GeoJSONLineStringGeometry;
}();
export { GeoJSONLineStringGeometry };
var GeoJSONRegion = /** @class */function (_super) {
  __extends(GeoJSONRegion, _super);
  function GeoJSONRegion(name, geometries, cp) {
    var _this = _super.call(this, name) || this;
    _this.type = 'geoJSON';
    _this.geometries = geometries;
    _this._center = cp && [cp[0], cp[1]];
    return _this;
  }
  GeoJSONRegion.prototype.calcCenter = function () {
    var geometries = this.geometries;
    var largestGeo;
    var largestGeoSize = 0;
    for (var i = 0; i < geometries.length; i++) {
      var geo = geometries[i];
      var exterior = geo.exterior;
      // Simple trick to use points count instead of polygon area as region size.
      // Ignore linestring
      var size = exterior && exterior.length;
      if (size > largestGeoSize) {
        largestGeo = geo;
        largestGeoSize = size;
      }
    }
    if (largestGeo) {
      return centroid(largestGeo.exterior);
    }
    // from bounding rect by default.
    var rect = this.getBoundingRect();
    return [rect.x + rect.width / 2, rect.y + rect.height / 2];
  };
  GeoJSONRegion.prototype.getBoundingRect = function (projection) {
    var rect = this._rect;
    // Always recalculate if using projection.
    if (rect && !projection) {
      return rect;
    }
    var min = [Infinity, Infinity];
    var max = [-Infinity, -Infinity];
    var geometries = this.geometries;
    each(geometries, function (geo) {
      if (geo.type === 'polygon') {
        // Doesn't consider hole
        updateBBoxFromPoints(geo.exterior, min, max, projection);
      } else {
        each(geo.points, function (points) {
          updateBBoxFromPoints(points, min, max, projection);
        });
      }
    });
    // Normalie invalid bounding.
    if (!(isFinite(min[0]) && isFinite(min[1]) && isFinite(max[0]) && isFinite(max[1]))) {
      min[0] = min[1] = max[0] = max[1] = 0;
    }
    rect = new BoundingRect(min[0], min[1], max[0] - min[0], max[1] - min[1]);
    if (!projection) {
      this._rect = rect;
    }
    return rect;
  };
  GeoJSONRegion.prototype.contain = function (coord) {
    var rect = this.getBoundingRect();
    var geometries = this.geometries;
    if (!rect.contain(coord[0], coord[1])) {
      return false;
    }
    loopGeo: for (var i = 0, len = geometries.length; i < len; i++) {
      var geo = geometries[i];
      // Only support polygon.
      if (geo.type !== 'polygon') {
        continue;
      }
      var exterior = geo.exterior;
      var interiors = geo.interiors;
      if (polygonContain.contain(exterior, coord[0], coord[1])) {
        // Not in the region if point is in the hole.
        for (var k = 0; k < (interiors ? interiors.length : 0); k++) {
          if (polygonContain.contain(interiors[k], coord[0], coord[1])) {
            continue loopGeo;
          }
        }
        return true;
      }
    }
    return false;
  };
  /**
   * Transform the raw coords to target bounding.
   * @param x
   * @param y
   * @param width
   * @param height
   */
  GeoJSONRegion.prototype.transformTo = function (x, y, width, height) {
    var rect = this.getBoundingRect();
    var aspect = rect.width / rect.height;
    if (!width) {
      width = aspect * height;
    } else if (!height) {
      height = width / aspect;
    }
    var target = new BoundingRect(x, y, width, height);
    var transform = rect.calculateTransform(target);
    var geometries = this.geometries;
    for (var i = 0; i < geometries.length; i++) {
      var geo = geometries[i];
      if (geo.type === 'polygon') {
        transformPoints(geo.exterior, transform);
        each(geo.interiors, function (interior) {
          transformPoints(interior, transform);
        });
      } else {
        each(geo.points, function (points) {
          transformPoints(points, transform);
        });
      }
    }
    rect = this._rect;
    rect.copy(target);
    // Update center
    this._center = [rect.x + rect.width / 2, rect.y + rect.height / 2];
  };
  GeoJSONRegion.prototype.cloneShallow = function (name) {
    name == null && (name = this.name);
    var newRegion = new GeoJSONRegion(name, this.geometries, this._center);
    newRegion._rect = this._rect;
    newRegion.transformTo = null; // Simply avoid to be called.
    return newRegion;
  };
  return GeoJSONRegion;
}(Region);
export { GeoJSONRegion };
var GeoSVGRegion = /** @class */function (_super) {
  __extends(GeoSVGRegion, _super);
  function GeoSVGRegion(name, elOnlyForCalculate) {
    var _this = _super.call(this, name) || this;
    _this.type = 'geoSVG';
    _this._elOnlyForCalculate = elOnlyForCalculate;
    return _this;
  }
  GeoSVGRegion.prototype.calcCenter = function () {
    var el = this._elOnlyForCalculate;
    var rect = el.getBoundingRect();
    var center = [rect.x + rect.width / 2, rect.y + rect.height / 2];
    var mat = matrix.identity(TMP_TRANSFORM);
    var target = el;
    while (target && !target.isGeoSVGGraphicRoot) {
      matrix.mul(mat, target.getLocalTransform(), mat);
      target = target.parent;
    }
    matrix.invert(mat, mat);
    vec2.applyTransform(center, center, mat);
    return center;
  };
  return GeoSVGRegion;
}(Region);
export { GeoSVGRegion };