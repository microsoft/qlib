
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
import RadiusAxis from './RadiusAxis.js';
import AngleAxis from './AngleAxis.js';
export var polarDimensions = ['radius', 'angle'];
var Polar = /** @class */function () {
  function Polar(name) {
    this.dimensions = polarDimensions;
    this.type = 'polar';
    /**
     * x of polar center
     */
    this.cx = 0;
    /**
     * y of polar center
     */
    this.cy = 0;
    this._radiusAxis = new RadiusAxis();
    this._angleAxis = new AngleAxis();
    this.axisPointerEnabled = true;
    this.name = name || '';
    this._radiusAxis.polar = this._angleAxis.polar = this;
  }
  /**
   * If contain coord
   */
  Polar.prototype.containPoint = function (point) {
    var coord = this.pointToCoord(point);
    return this._radiusAxis.contain(coord[0]) && this._angleAxis.contain(coord[1]);
  };
  /**
   * If contain data
   */
  Polar.prototype.containData = function (data) {
    return this._radiusAxis.containData(data[0]) && this._angleAxis.containData(data[1]);
  };
  Polar.prototype.getAxis = function (dim) {
    var key = '_' + dim + 'Axis';
    return this[key];
  };
  Polar.prototype.getAxes = function () {
    return [this._radiusAxis, this._angleAxis];
  };
  /**
   * Get axes by type of scale
   */
  Polar.prototype.getAxesByScale = function (scaleType) {
    var axes = [];
    var angleAxis = this._angleAxis;
    var radiusAxis = this._radiusAxis;
    angleAxis.scale.type === scaleType && axes.push(angleAxis);
    radiusAxis.scale.type === scaleType && axes.push(radiusAxis);
    return axes;
  };
  Polar.prototype.getAngleAxis = function () {
    return this._angleAxis;
  };
  Polar.prototype.getRadiusAxis = function () {
    return this._radiusAxis;
  };
  Polar.prototype.getOtherAxis = function (axis) {
    var angleAxis = this._angleAxis;
    return axis === angleAxis ? this._radiusAxis : angleAxis;
  };
  /**
   * Base axis will be used on stacking.
   *
   */
  Polar.prototype.getBaseAxis = function () {
    return this.getAxesByScale('ordinal')[0] || this.getAxesByScale('time')[0] || this.getAngleAxis();
  };
  Polar.prototype.getTooltipAxes = function (dim) {
    var baseAxis = dim != null && dim !== 'auto' ? this.getAxis(dim) : this.getBaseAxis();
    return {
      baseAxes: [baseAxis],
      otherAxes: [this.getOtherAxis(baseAxis)]
    };
  };
  /**
   * Convert a single data item to (x, y) point.
   * Parameter data is an array which the first element is radius and the second is angle
   */
  Polar.prototype.dataToPoint = function (data, clamp) {
    return this.coordToPoint([this._radiusAxis.dataToRadius(data[0], clamp), this._angleAxis.dataToAngle(data[1], clamp)]);
  };
  /**
   * Convert a (x, y) point to data
   */
  Polar.prototype.pointToData = function (point, clamp) {
    var coord = this.pointToCoord(point);
    return [this._radiusAxis.radiusToData(coord[0], clamp), this._angleAxis.angleToData(coord[1], clamp)];
  };
  /**
   * Convert a (x, y) point to (radius, angle) coord
   */
  Polar.prototype.pointToCoord = function (point) {
    var dx = point[0] - this.cx;
    var dy = point[1] - this.cy;
    var angleAxis = this.getAngleAxis();
    var extent = angleAxis.getExtent();
    var minAngle = Math.min(extent[0], extent[1]);
    var maxAngle = Math.max(extent[0], extent[1]);
    // Fix fixed extent in polarCreator
    // FIXME
    angleAxis.inverse ? minAngle = maxAngle - 360 : maxAngle = minAngle + 360;
    var radius = Math.sqrt(dx * dx + dy * dy);
    dx /= radius;
    dy /= radius;
    var radian = Math.atan2(-dy, dx) / Math.PI * 180;
    // move to angleExtent
    var dir = radian < minAngle ? 1 : -1;
    while (radian < minAngle || radian > maxAngle) {
      radian += dir * 360;
    }
    return [radius, radian];
  };
  /**
   * Convert a (radius, angle) coord to (x, y) point
   */
  Polar.prototype.coordToPoint = function (coord) {
    var radius = coord[0];
    var radian = coord[1] / 180 * Math.PI;
    var x = Math.cos(radian) * radius + this.cx;
    // Inverse the y
    var y = -Math.sin(radian) * radius + this.cy;
    return [x, y];
  };
  /**
   * Get ring area of cartesian.
   * Area will have a contain function to determine if a point is in the coordinate system.
   */
  Polar.prototype.getArea = function () {
    var angleAxis = this.getAngleAxis();
    var radiusAxis = this.getRadiusAxis();
    var radiusExtent = radiusAxis.getExtent().slice();
    radiusExtent[0] > radiusExtent[1] && radiusExtent.reverse();
    var angleExtent = angleAxis.getExtent();
    var RADIAN = Math.PI / 180;
    var EPSILON = 1e-4;
    return {
      cx: this.cx,
      cy: this.cy,
      r0: radiusExtent[0],
      r: radiusExtent[1],
      startAngle: -angleExtent[0] * RADIAN,
      endAngle: -angleExtent[1] * RADIAN,
      clockwise: angleAxis.inverse,
      contain: function (x, y) {
        // It's a ring shape.
        // Start angle and end angle don't matter
        var dx = x - this.cx;
        var dy = y - this.cy;
        var d2 = dx * dx + dy * dy;
        var r = this.r;
        var r0 = this.r0;
        // minus a tiny value 1e-4 in double side to avoid being clipped unexpectedly
        // r == r0 contain nothing
        return r !== r0 && d2 - EPSILON <= r * r && d2 + EPSILON >= r0 * r0;
      }
    };
  };
  Polar.prototype.convertToPixel = function (ecModel, finder, value) {
    var coordSys = getCoordSys(finder);
    return coordSys === this ? this.dataToPoint(value) : null;
  };
  Polar.prototype.convertFromPixel = function (ecModel, finder, pixel) {
    var coordSys = getCoordSys(finder);
    return coordSys === this ? this.pointToData(pixel) : null;
  };
  return Polar;
}();
function getCoordSys(finder) {
  var seriesModel = finder.seriesModel;
  var polarModel = finder.polarModel;
  return polarModel && polarModel.coordinateSystem || seriesModel && seriesModel.coordinateSystem;
}
export default Polar;