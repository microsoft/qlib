
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
/**
 * Grid is a region which contains at most 4 cartesian systems
 *
 * TODO Default cartesian
 */
import { isObject, each, indexOf, retrieve3, keys } from 'zrender/lib/core/util.js';
import { getLayoutRect } from '../../util/layout.js';
import { createScaleByModel, ifAxisCrossZero, niceScaleExtent, estimateLabelUnionRect, getDataDimensionsOnAxis } from '../../coord/axisHelper.js';
import Cartesian2D, { cartesian2DDimensions } from './Cartesian2D.js';
import Axis2D from './Axis2D.js';
import { SINGLE_REFERRING } from '../../util/model.js';
import { isCartesian2DSeries, findAxisModels } from './cartesianAxisHelper.js';
import { isIntervalOrLogScale } from '../../scale/helper.js';
import { alignScaleTicks } from '../axisAlignTicks.js';
var Grid = /** @class */function () {
  function Grid(gridModel, ecModel, api) {
    // FIXME:TS where used (different from registered type 'cartesian2d')?
    this.type = 'grid';
    this._coordsMap = {};
    this._coordsList = [];
    this._axesMap = {};
    this._axesList = [];
    this.axisPointerEnabled = true;
    this.dimensions = cartesian2DDimensions;
    this._initCartesian(gridModel, ecModel, api);
    this.model = gridModel;
  }
  Grid.prototype.getRect = function () {
    return this._rect;
  };
  Grid.prototype.update = function (ecModel, api) {
    var axesMap = this._axesMap;
    this._updateScale(ecModel, this.model);
    function updateAxisTicks(axes) {
      var alignTo;
      // Axis is added in order of axisIndex.
      var axesIndices = keys(axes);
      var len = axesIndices.length;
      if (!len) {
        return;
      }
      var axisNeedsAlign = [];
      // Process once and calculate the ticks for those don't use alignTicks.
      for (var i = len - 1; i >= 0; i--) {
        var idx = +axesIndices[i]; // Convert to number.
        var axis = axes[idx];
        var model = axis.model;
        var scale = axis.scale;
        if (
        // Only value and log axis without interval support alignTicks.
        isIntervalOrLogScale(scale) && model.get('alignTicks') && model.get('interval') == null) {
          axisNeedsAlign.push(axis);
        } else {
          niceScaleExtent(scale, model);
          if (isIntervalOrLogScale(scale)) {
            // Can only align to interval or log axis.
            alignTo = axis;
          }
        }
      }
      ;
      // All axes has set alignTicks. Pick the first one.
      // PENDING. Should we find the axis that both set interval, min, max and align to this one?
      if (axisNeedsAlign.length) {
        if (!alignTo) {
          alignTo = axisNeedsAlign.pop();
          niceScaleExtent(alignTo.scale, alignTo.model);
        }
        each(axisNeedsAlign, function (axis) {
          alignScaleTicks(axis.scale, axis.model, alignTo.scale);
        });
      }
    }
    updateAxisTicks(axesMap.x);
    updateAxisTicks(axesMap.y);
    // Key: axisDim_axisIndex, value: boolean, whether onZero target.
    var onZeroRecords = {};
    each(axesMap.x, function (xAxis) {
      fixAxisOnZero(axesMap, 'y', xAxis, onZeroRecords);
    });
    each(axesMap.y, function (yAxis) {
      fixAxisOnZero(axesMap, 'x', yAxis, onZeroRecords);
    });
    // Resize again if containLabel is enabled
    // FIXME It may cause getting wrong grid size in data processing stage
    this.resize(this.model, api);
  };
  /**
   * Resize the grid
   */
  Grid.prototype.resize = function (gridModel, api, ignoreContainLabel) {
    var boxLayoutParams = gridModel.getBoxLayoutParams();
    var isContainLabel = !ignoreContainLabel && gridModel.get('containLabel');
    var gridRect = getLayoutRect(boxLayoutParams, {
      width: api.getWidth(),
      height: api.getHeight()
    });
    this._rect = gridRect;
    var axesList = this._axesList;
    adjustAxes();
    // Minus label size
    if (isContainLabel) {
      each(axesList, function (axis) {
        if (!axis.model.get(['axisLabel', 'inside'])) {
          var labelUnionRect = estimateLabelUnionRect(axis);
          if (labelUnionRect) {
            var dim = axis.isHorizontal() ? 'height' : 'width';
            var margin = axis.model.get(['axisLabel', 'margin']);
            gridRect[dim] -= labelUnionRect[dim] + margin;
            if (axis.position === 'top') {
              gridRect.y += labelUnionRect.height + margin;
            } else if (axis.position === 'left') {
              gridRect.x += labelUnionRect.width + margin;
            }
          }
        }
      });
      adjustAxes();
    }
    each(this._coordsList, function (coord) {
      // Calculate affine matrix to accelerate the data to point transform.
      // If all the axes scales are time or value.
      coord.calcAffineTransform();
    });
    function adjustAxes() {
      each(axesList, function (axis) {
        var isHorizontal = axis.isHorizontal();
        var extent = isHorizontal ? [0, gridRect.width] : [0, gridRect.height];
        var idx = axis.inverse ? 1 : 0;
        axis.setExtent(extent[idx], extent[1 - idx]);
        updateAxisTransform(axis, isHorizontal ? gridRect.x : gridRect.y);
      });
    }
  };
  Grid.prototype.getAxis = function (dim, axisIndex) {
    var axesMapOnDim = this._axesMap[dim];
    if (axesMapOnDim != null) {
      return axesMapOnDim[axisIndex || 0];
    }
  };
  Grid.prototype.getAxes = function () {
    return this._axesList.slice();
  };
  Grid.prototype.getCartesian = function (xAxisIndex, yAxisIndex) {
    if (xAxisIndex != null && yAxisIndex != null) {
      var key = 'x' + xAxisIndex + 'y' + yAxisIndex;
      return this._coordsMap[key];
    }
    if (isObject(xAxisIndex)) {
      yAxisIndex = xAxisIndex.yAxisIndex;
      xAxisIndex = xAxisIndex.xAxisIndex;
    }
    for (var i = 0, coordList = this._coordsList; i < coordList.length; i++) {
      if (coordList[i].getAxis('x').index === xAxisIndex || coordList[i].getAxis('y').index === yAxisIndex) {
        return coordList[i];
      }
    }
  };
  Grid.prototype.getCartesians = function () {
    return this._coordsList.slice();
  };
  /**
   * @implements
   */
  Grid.prototype.convertToPixel = function (ecModel, finder, value) {
    var target = this._findConvertTarget(finder);
    return target.cartesian ? target.cartesian.dataToPoint(value) : target.axis ? target.axis.toGlobalCoord(target.axis.dataToCoord(value)) : null;
  };
  /**
   * @implements
   */
  Grid.prototype.convertFromPixel = function (ecModel, finder, value) {
    var target = this._findConvertTarget(finder);
    return target.cartesian ? target.cartesian.pointToData(value) : target.axis ? target.axis.coordToData(target.axis.toLocalCoord(value)) : null;
  };
  Grid.prototype._findConvertTarget = function (finder) {
    var seriesModel = finder.seriesModel;
    var xAxisModel = finder.xAxisModel || seriesModel && seriesModel.getReferringComponents('xAxis', SINGLE_REFERRING).models[0];
    var yAxisModel = finder.yAxisModel || seriesModel && seriesModel.getReferringComponents('yAxis', SINGLE_REFERRING).models[0];
    var gridModel = finder.gridModel;
    var coordsList = this._coordsList;
    var cartesian;
    var axis;
    if (seriesModel) {
      cartesian = seriesModel.coordinateSystem;
      indexOf(coordsList, cartesian) < 0 && (cartesian = null);
    } else if (xAxisModel && yAxisModel) {
      cartesian = this.getCartesian(xAxisModel.componentIndex, yAxisModel.componentIndex);
    } else if (xAxisModel) {
      axis = this.getAxis('x', xAxisModel.componentIndex);
    } else if (yAxisModel) {
      axis = this.getAxis('y', yAxisModel.componentIndex);
    }
    // Lowest priority.
    else if (gridModel) {
      var grid = gridModel.coordinateSystem;
      if (grid === this) {
        cartesian = this._coordsList[0];
      }
    }
    return {
      cartesian: cartesian,
      axis: axis
    };
  };
  /**
   * @implements
   */
  Grid.prototype.containPoint = function (point) {
    var coord = this._coordsList[0];
    if (coord) {
      return coord.containPoint(point);
    }
  };
  /**
   * Initialize cartesian coordinate systems
   */
  Grid.prototype._initCartesian = function (gridModel, ecModel, api) {
    var _this = this;
    var grid = this;
    var axisPositionUsed = {
      left: false,
      right: false,
      top: false,
      bottom: false
    };
    var axesMap = {
      x: {},
      y: {}
    };
    var axesCount = {
      x: 0,
      y: 0
    };
    // Create axis
    ecModel.eachComponent('xAxis', createAxisCreator('x'), this);
    ecModel.eachComponent('yAxis', createAxisCreator('y'), this);
    if (!axesCount.x || !axesCount.y) {
      // Roll back when there no either x or y axis
      this._axesMap = {};
      this._axesList = [];
      return;
    }
    this._axesMap = axesMap;
    // Create cartesian2d
    each(axesMap.x, function (xAxis, xAxisIndex) {
      each(axesMap.y, function (yAxis, yAxisIndex) {
        var key = 'x' + xAxisIndex + 'y' + yAxisIndex;
        var cartesian = new Cartesian2D(key);
        cartesian.master = _this;
        cartesian.model = gridModel;
        _this._coordsMap[key] = cartesian;
        _this._coordsList.push(cartesian);
        cartesian.addAxis(xAxis);
        cartesian.addAxis(yAxis);
      });
    });
    function createAxisCreator(dimName) {
      return function (axisModel, idx) {
        if (!isAxisUsedInTheGrid(axisModel, gridModel)) {
          return;
        }
        var axisPosition = axisModel.get('position');
        if (dimName === 'x') {
          // Fix position
          if (axisPosition !== 'top' && axisPosition !== 'bottom') {
            // Default bottom of X
            axisPosition = axisPositionUsed.bottom ? 'top' : 'bottom';
          }
        } else {
          // Fix position
          if (axisPosition !== 'left' && axisPosition !== 'right') {
            // Default left of Y
            axisPosition = axisPositionUsed.left ? 'right' : 'left';
          }
        }
        axisPositionUsed[axisPosition] = true;
        var axis = new Axis2D(dimName, createScaleByModel(axisModel), [0, 0], axisModel.get('type'), axisPosition);
        var isCategory = axis.type === 'category';
        axis.onBand = isCategory && axisModel.get('boundaryGap');
        axis.inverse = axisModel.get('inverse');
        // Inject axis into axisModel
        axisModel.axis = axis;
        // Inject axisModel into axis
        axis.model = axisModel;
        // Inject grid info axis
        axis.grid = grid;
        // Index of axis, can be used as key
        axis.index = idx;
        grid._axesList.push(axis);
        axesMap[dimName][idx] = axis;
        axesCount[dimName]++;
      };
    }
  };
  /**
   * Update cartesian properties from series.
   */
  Grid.prototype._updateScale = function (ecModel, gridModel) {
    // Reset scale
    each(this._axesList, function (axis) {
      axis.scale.setExtent(Infinity, -Infinity);
      if (axis.type === 'category') {
        var categorySortInfo = axis.model.get('categorySortInfo');
        axis.scale.setSortInfo(categorySortInfo);
      }
    });
    ecModel.eachSeries(function (seriesModel) {
      if (isCartesian2DSeries(seriesModel)) {
        var axesModelMap = findAxisModels(seriesModel);
        var xAxisModel = axesModelMap.xAxisModel;
        var yAxisModel = axesModelMap.yAxisModel;
        if (!isAxisUsedInTheGrid(xAxisModel, gridModel) || !isAxisUsedInTheGrid(yAxisModel, gridModel)) {
          return;
        }
        var cartesian = this.getCartesian(xAxisModel.componentIndex, yAxisModel.componentIndex);
        var data = seriesModel.getData();
        var xAxis = cartesian.getAxis('x');
        var yAxis = cartesian.getAxis('y');
        unionExtent(data, xAxis);
        unionExtent(data, yAxis);
      }
    }, this);
    function unionExtent(data, axis) {
      each(getDataDimensionsOnAxis(data, axis.dim), function (dim) {
        axis.scale.unionExtentFromData(data, dim);
      });
    }
  };
  /**
   * @param dim 'x' or 'y' or 'auto' or null/undefined
   */
  Grid.prototype.getTooltipAxes = function (dim) {
    var baseAxes = [];
    var otherAxes = [];
    each(this.getCartesians(), function (cartesian) {
      var baseAxis = dim != null && dim !== 'auto' ? cartesian.getAxis(dim) : cartesian.getBaseAxis();
      var otherAxis = cartesian.getOtherAxis(baseAxis);
      indexOf(baseAxes, baseAxis) < 0 && baseAxes.push(baseAxis);
      indexOf(otherAxes, otherAxis) < 0 && otherAxes.push(otherAxis);
    });
    return {
      baseAxes: baseAxes,
      otherAxes: otherAxes
    };
  };
  Grid.create = function (ecModel, api) {
    var grids = [];
    ecModel.eachComponent('grid', function (gridModel, idx) {
      var grid = new Grid(gridModel, ecModel, api);
      grid.name = 'grid_' + idx;
      // dataSampling requires axis extent, so resize
      // should be performed in create stage.
      grid.resize(gridModel, api, true);
      gridModel.coordinateSystem = grid;
      grids.push(grid);
    });
    // Inject the coordinateSystems into seriesModel
    ecModel.eachSeries(function (seriesModel) {
      if (!isCartesian2DSeries(seriesModel)) {
        return;
      }
      var axesModelMap = findAxisModels(seriesModel);
      var xAxisModel = axesModelMap.xAxisModel;
      var yAxisModel = axesModelMap.yAxisModel;
      var gridModel = xAxisModel.getCoordSysModel();
      if (process.env.NODE_ENV !== 'production') {
        if (!gridModel) {
          throw new Error('Grid "' + retrieve3(xAxisModel.get('gridIndex'), xAxisModel.get('gridId'), 0) + '" not found');
        }
        if (xAxisModel.getCoordSysModel() !== yAxisModel.getCoordSysModel()) {
          throw new Error('xAxis and yAxis must use the same grid');
        }
      }
      var grid = gridModel.coordinateSystem;
      seriesModel.coordinateSystem = grid.getCartesian(xAxisModel.componentIndex, yAxisModel.componentIndex);
    });
    return grids;
  };
  // For deciding which dimensions to use when creating list data
  Grid.dimensions = cartesian2DDimensions;
  return Grid;
}();
/**
 * Check if the axis is used in the specified grid.
 */
function isAxisUsedInTheGrid(axisModel, gridModel) {
  return axisModel.getCoordSysModel() === gridModel;
}
function fixAxisOnZero(axesMap, otherAxisDim, axis,
// Key: see `getOnZeroRecordKey`
onZeroRecords) {
  axis.getAxesOnZeroOf = function () {
    // TODO: onZero of multiple axes.
    return otherAxisOnZeroOf ? [otherAxisOnZeroOf] : [];
  };
  // onZero can not be enabled in these two situations:
  // 1. When any other axis is a category axis.
  // 2. When no axis is cross 0 point.
  var otherAxes = axesMap[otherAxisDim];
  var otherAxisOnZeroOf;
  var axisModel = axis.model;
  var onZero = axisModel.get(['axisLine', 'onZero']);
  var onZeroAxisIndex = axisModel.get(['axisLine', 'onZeroAxisIndex']);
  if (!onZero) {
    return;
  }
  // If target axis is specified.
  if (onZeroAxisIndex != null) {
    if (canOnZeroToAxis(otherAxes[onZeroAxisIndex])) {
      otherAxisOnZeroOf = otherAxes[onZeroAxisIndex];
    }
  } else {
    // Find the first available other axis.
    for (var idx in otherAxes) {
      if (otherAxes.hasOwnProperty(idx) && canOnZeroToAxis(otherAxes[idx])
      // Consider that two Y axes on one value axis,
      // if both onZero, the two Y axes overlap.
      && !onZeroRecords[getOnZeroRecordKey(otherAxes[idx])]) {
        otherAxisOnZeroOf = otherAxes[idx];
        break;
      }
    }
  }
  if (otherAxisOnZeroOf) {
    onZeroRecords[getOnZeroRecordKey(otherAxisOnZeroOf)] = true;
  }
  function getOnZeroRecordKey(axis) {
    return axis.dim + '_' + axis.index;
  }
}
function canOnZeroToAxis(axis) {
  return axis && axis.type !== 'category' && axis.type !== 'time' && ifAxisCrossZero(axis);
}
function updateAxisTransform(axis, coordBase) {
  var axisExtent = axis.getExtent();
  var axisExtentSum = axisExtent[0] + axisExtent[1];
  // Fast transform
  axis.toGlobalCoord = axis.dim === 'x' ? function (coord) {
    return coord + coordBase;
  } : function (coord) {
    return axisExtentSum - coord + coordBase;
  };
  axis.toLocalCoord = axis.dim === 'x' ? function (coord) {
    return coord - coordBase;
  } : function (coord) {
    return axisExtentSum - coord + coordBase;
  };
}
export default Grid;